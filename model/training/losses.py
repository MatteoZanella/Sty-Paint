import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
import torchvision
from model.networks.light_renderer import LightRenderer

class KLDivergence(nn.Module):

    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, mu, log_sigma):
        kl_loss = 1 + log_sigma - mu.pow(2) - log_sigma.exp()   # bs x n_features
        kl_loss = -0.5 * torch.sum(kl_loss, dim=-1)
        kl_loss = torch.mean(kl_loss)

        return dict(kl_div=kl_loss)

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


########################################################################################################################
class MSECalculator :

    def __init__(self, loss_weights) :
        self.MSE = nn.MSELoss(reduction='mean')

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
        mse_position = self.MSE(preds_position, target_position)
        mse_size = self.MSE(preds_size, target_size)
        mse_theta = self.MSE(preds_theta, target_theta)
        mse_color = self.MSE(preds_color, target_color)

        #
        losses = dict(
            mse_position = mse_position,
            mse_size = mse_size,
            mse_theta = mse_theta,
            mse_color = mse_color)

        return losses

########################################################################################################################
class ReferenceImageLoss:

    def __init__(self, render=False, meta_brushes_path=None, canvas_size=None):

        self.render = render # Set to render the strokes during training

        if not self.render:
            self.blur = torchvision.transforms.GaussianBlur(kernel_size=(7, 7))
            reduction = 'mean'
        else:
            self.renderer = LightRenderer(brushes_path=meta_brushes_path, canvas_size=canvas_size)
            reduction = 'none'

        self.MSE = nn.MSELoss(reduction=reduction)
    def __call__(self, predictions, ref_imgs, isEval=False):

        if not self.render:
            preds_position = predictions[:, :, :2]
            preds_color = predictions[:, :, 5:]

            ref_imgs = self.blur(ref_imgs)
            ref_imgs = repeat(ref_imgs, 'bs ch h w -> (bs L) ch h w', L=predictions.size(1))
            grid = rearrange(preds_position, 'bs L dim -> (bs L) 1 1 dim')
            target_color_img = F.grid_sample(ref_imgs, 2 * grid - 1, align_corners=False)
            target_color_img = rearrange(target_color_img, '(bs L) ch 1 1 -> bs L ch', L=predictions.size(1))

            loss = self.MSE(preds_color, target_color_img) * self.weight
        else:
            ref_imgs = repeat(ref_imgs, 'bs ch h w -> bs L ch h w', L=predictions.size(1))
            strokes, area = self.renderer(predictions)
            loss = torch.sum(self.MSE(strokes, ref_imgs), dim=[2,3,4]) / torch.sum(area, dim=[2,3,4])
            loss = torch.mean(loss) * self.weight

        return dict(mse_color_img=loss)
