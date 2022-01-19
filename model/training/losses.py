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
        return target_tensor.expand_as(prediction).to(prediction.device)

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

def cal_gradient_penalty(netD, real_data, fake_data, device, context = None, type='mixed', constant=1.0, lambda_gp=10.0):
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
        disc_interpolates = netD(interpolatesv, context=context)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].reshape(real_data.size(0), -1)  # flat the data
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
class ColorImageLoss(nn.Module):
    def __init__(self, mode='l2'):
        super(ColorImageLoss, self).__init__()
        self.blur = torchvision.transforms.GaussianBlur(kernel_size=(7, 7))

        # Criterion
        if mode == 'l2' :
            self.criterion = nn.MSELoss()
        elif mode == 'l1' :
            self.criterion = nn.L1Loss()
        else :
            raise NotImplementedError()

    def __call__(self, predictions, ref_imgs):

        preds_position = predictions[:, :, :2]
        preds_color = predictions[:, :, 5:]

        ref_imgs = self.blur(ref_imgs)
        ref_imgs = repeat(ref_imgs, 'bs ch h w -> (bs L) ch h w', L=predictions.size(1))
        grid = rearrange(preds_position, 'bs L dim -> (bs L) 1 1 dim').detach()
        target_color_img = F.grid_sample(ref_imgs, 2 * grid - 1, align_corners=False, padding_mode='border')
        target_color_img = rearrange(target_color_img, '(bs L) ch 1 1 -> bs L ch', L=predictions.size(1))
        loss = self.criterion(preds_color, target_color_img)

        return loss


######################################
class RenderImageLoss(nn.Module):
    def __init__(self, config):
        super(RenderImageLoss, self).__init__()

        self.type = config["train"]["losses"]["reference_img"]["render"]["type"]
        self.active = config["train"]["losses"]["reference_img"]["render"]["weight"] > 0
        mode = config["train"]["losses"]["reference_img"]["render"]["mode"]
        meta_brushes_path = config["renderer"]["brushes_path"]
        canvas_size = config["dataset"]["resize"]

        if self.type == 'full':
            print('Full loss')
            # render the strokes
            self.renderer = LightRenderer(brushes_path=meta_brushes_path, canvas_size=canvas_size)
            # reduction = 'none'
            reduction = 'mean'
        elif self.type == 'masked':
            print('masked')
            # render the strokes
            self.renderer = LightRenderer(brushes_path=meta_brushes_path, canvas_size=canvas_size)
            # reduction = 'none'
            reduction = 'none'
        else:
            raise NotImplementedError()

        # Criterion
        if mode == 'l2' :
            self.criterion = nn.MSELoss(reduction=reduction)
        elif mode == 'l1' :
            self.criterion = nn.L1Loss(reduction=reduction)
        else :
            raise NotImplementedError()

    def __call__(self, predictions, ref_imgs, canvas_start=None):

        if not self.active:
            return torch.tensor([0], device=predictions.device)

        if self.type == 'full':
            rec = self.renderer(predictions, canvas_start)
            loss = self.criterion(rec, ref_imgs)
            if not math.isfinite(loss):
                sys.exit('Loss is nan, stop training')
            else:
                return loss
        else:
            ref_imgs = repeat(ref_imgs, 'bs ch h w -> bs L ch h w', L=predictions.size(1))
            foregrounds, alphas = self.renderer.render_single_strokes(predictions)
            loss = self.criterion(foregrounds, ref_imgs) * alphas
            loss = loss.mean()

            return loss
##################################
class PosColorLoss(nn.Module):
    def __init__(self, mode='l2'):
        super(PosColorLoss, self).__init__()
        self.blur = torchvision.transforms.GaussianBlur(kernel_size=(7, 7))

        # Criterion
        if mode == 'l2' :
            self.criterion = nn.MSELoss()
        elif mode == 'l1' :
            self.criterion = nn.L1Loss()
        else :
            raise NotImplementedError()

    def __call__(self, predictions, ref_imgs):

        preds_position = predictions[:, :, :2]

        ref_imgs = self.blur(ref_imgs)
        ref_imgs = repeat(ref_imgs, 'bs ch h w -> (bs L) ch h w', L=predictions.size(1))
        grid = rearrange(preds_position, 'bs L dim -> (bs L) 1 1 dim')
        gt_color = F.grid_sample(ref_imgs, 2 * grid - 1, align_corners=False, padding_mode='border')
        gt_color = rearrange(gt_color, '(bs L) ch 1 1 -> bs L ch', L=predictions.size(1))
        loss = torch.pow(torch.diff(gt_color, dim=1), 2).mean()
        return loss


########################################################################
class DistLoss(nn.Module) :
    def __init__(self, mode='l2') :
        '''
        Args:
            mode:

        Aim of this loss is to change the position of the strokes such that subsequent strokes have similar color.
        To do this, we:
        1. Find the color of the reference image for each predicted stroke
        2. Find the topK corespondencies with that color, and the respective (tgt_x, tgt_y) locations
        3. Compute the distance between the current predicted position and the closest postion among the ones found by
        the step 2. for the stroke before (or after) in the sequence
        '''
        super(DistLoss, self).__init__()
        # Criterion
        if mode == 'l2' :
            self.p = 2
        elif mode == 'l1' :
            self.p = 1
        else :
            raise NotImplementedError()

        self.K = 8
        print(f'Using {self.K} NN')

    def __call__(self, predictions, ref_imgs) :

        bs, L,_ = predictions.shape
        img_size = ref_imgs.shape[-1]

        preds_position = predictions[:, :, :2]

        # Sample GT colors
        feat_temp = repeat(ref_imgs, 'bs ch h w -> (L bs) ch h w', L=L)
        grid = rearrange(preds_position, 'L bs p -> (L bs) 1 1 p').detach()
        pooled_colors = F.grid_sample(feat_temp, 2 * grid - 1, align_corners=False, mode='nearest',
                                      padding_mode='border')

        feat_temp = rearrange(feat_temp, '(L bs) ch h w -> bs L ch h w', bs=bs, L=L)
        pooled_colors = rearrange(pooled_colors, '(L bs) ch 1 1 -> bs L ch 1 1', bs=bs, L=L)

        # Find top K similar color across the image
        similar_colors = torch.abs(feat_temp - pooled_colors).mean(dim=2)  # average over ch
        _, idx = torch.topk(-similar_colors.reshape(bs, L, img_size * img_size), k=self.K, dim=-1)

        # Convert idx to (x,y)
        tgt_y = torch.div(idx, img_size, rounding_mode='floor') / img_size
        tgt_x = torch.remainder(idx, img_size) / img_size

        tgt = torch.stack([tgt_x.reshape(bs, L * self.K), tgt_y.reshape(bs, L * self.K)], dim=-1)
        tgt = tgt.reshape(bs, L, self.K, 2)


        # Shift the target up and down by 1 position, i.e. compare each element of the sequence with the precedent
        # and the following one
        tgt_down = torch.roll(tgt, shifts=1, dims=1)
        #tgt_up = torch.roll(tgt, shifts=-1, dims=1)

        # Computes distances
        preds_position = repeat(preds_position, 'bs L p -> bs L K p', K=self.K)
        dist_down = torch.cdist(preds_position, tgt_down, p=self.p)
        #dist_up = torch.cdist(preds_position, tgt_up, p=self.p)

        # Find the closest point, and use it to compute the loss
        val_down, _ = torch.min(dist_down, dim=-1)
        #val_up, _ = torch.min(dist_up, dim=-1)

        # Take care of first and last element of the sequence, should not be compared
        #loss = torch.mean(0.5 * torch.add(val_up[:, 1:], val_down[:, :-1]))
        loss = torch.mean(val_down[:, 1:])

        return loss