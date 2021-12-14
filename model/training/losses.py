import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
import torchvision
from model.networks.light_renderer import LightRenderer
import math
import sys

class KLDivergence(nn.Module):

    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, mu, log_sigma):
        kl_loss = 1 + log_sigma - mu.pow(2) - log_sigma.exp()   # bs x n_features
        kl_loss = -0.5 * torch.sum(kl_loss, dim=-1)
        kl_loss = torch.mean(kl_loss)

        return kl_loss

########################################################################################################################

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


########################################################################################################################
class ReconstructionLoss(nn.Module):

    def __init__(self, mode='l2') :
        super(ReconstructionLoss, self).__init__()

        if mode == 'l2':
            self.criterion = nn.MSELoss(reduction='mean')
        elif mode == 'l1':
            self.criterion = nn.L1Loss(reduction='mean')
        else:
            raise NotImplementedError()
    def __call__(self,
                 predictions,
                 targets,
                 isEval=False) :
        '''
        Args:
            predictions: [bs x L x dim] predicted sequence
            targets: [bs x L x dim] target sequence
        Returns:
            loss
        '''
        # Slice prediction
        preds_position = predictions[:, :, :2]
        preds_size = predictions[:, :, 2:4]
        preds_theta = predictions[:, :, 4]
        preds_color = predictions[:, :, 5:]

        # Slice target
        target_position = targets[:, :, :2]
        target_size = targets[:, :, 2:4]
        target_theta = targets[:, :, 4]
        target_color = targets[:, :, 5:]

        # Compute losses
        mse_position = self.criterion(preds_position, target_position)
        mse_size = self.criterion(preds_size, target_size)
        mse_theta = self.criterion(preds_theta, target_theta)
        mse_color = self.criterion(preds_color, target_color)

        return mse_position, mse_size, mse_theta, mse_color

########################################################################################################################
class ReferenceImageLoss(nn.Module):
    def __init__(self, mode='l2', use_renderer=True, meta_brushes_path=None, canvas_size=None):
        super(ReferenceImageLoss, self).__init__()

        self.use_renderer = use_renderer
        if not self.use_renderer:
            # use the color of the reference image underlying the stroke
            self.blur = torchvision.transforms.GaussianBlur(kernel_size=(7, 7))
            reduction = 'mean'
        else:
            # render the strokes
            self.renderer = LightRenderer(brushes_path=meta_brushes_path, canvas_size=canvas_size)
            reduction = 'none'

        # Criterion
        if mode == 'l2' :
            self.criterion = nn.MSELoss(reduction=reduction)
        elif mode == 'l1' :
            self.criterion = nn.L1Loss(reduction=reduction)
        else :
            raise NotImplementedError()

    def __call__(self, predictions, ref_imgs):

        if not self.use_renderer:
            preds_position = predictions[:, :, :2]
            preds_color = predictions[:, :, 5:]

            ref_imgs = self.blur(ref_imgs)
            ref_imgs = repeat(ref_imgs, 'bs ch h w -> (bs L) ch h w', L=predictions.size(1))
            grid = rearrange(preds_position, 'bs L dim -> (bs L) 1 1 dim')
            target_color_img = F.grid_sample(ref_imgs, 2 * grid - 1, align_corners=False)
            target_color_img = rearrange(target_color_img, '(bs L) ch 1 1 -> bs L ch', L=predictions.size(1))
            loss = self.criterion(preds_color, target_color_img)
            return loss
        else:
            ref_imgs = torch.repeat_interleave(ref_imgs, repeats=predictions.size(1), dim=0)
            brush, alphas = self.renderer(predictions)

            area = alphas.sum(dim=[2, 3], keepdim=True) # sum over spatial dimensions
            area = torch.clamp(area, min=1) # TODO: fix here
            loss = self.criterion(brush, ref_imgs) * alphas
            loss = loss.sum(dim=[2, 3]) / area   # sum over spatial dimension and normalize by area

            loss = loss.mean()   # average over channels and batch size
            if not math.isfinite(loss):
                sys.exit('Loss is nan, stop training')
            else:
                return loss
