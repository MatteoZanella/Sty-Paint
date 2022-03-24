import math
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
import torchvision
from model.networks.light_renderer import LightRenderer

# ======================================================================================================================
# KL Divergence
class KLDivergence(nn.Module):

    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, mu, log_sigma):
        kl_loss = 1 + log_sigma - mu.pow(2) - log_sigma.exp()  # bs x n_features
        kl_loss = -0.5 * torch.sum(kl_loss, dim=-1)
        kl_loss = torch.mean(kl_loss)

        return kl_loss


# ======================================================================================================================
# GAN Losses form:
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


def cal_gradient_penalty(netD, real_data, fake_data, device, context=None, type='mixed', constant=1.0, lambda_gp=10.0):
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
        if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                *real_data.shape)
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
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


# ======================================================================================================================
# L2 loss on strokes parameters
class ReconstructionLoss(nn.Module):

    def __init__(self, mode='l2'):
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
                 isEval=False):
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


# ======================================================================================================================
class ColorImageLoss(nn.Module):
    def __init__(self, config):
        super(ColorImageLoss, self).__init__()
        self.blur = torchvision.transforms.GaussianBlur(kernel_size=(7, 7))
        mode = config["train"]["losses"]["reference_img"]["color"]["mode"]
        self.detach_grid = config["train"]["losses"]["reference_img"]["color"]["detach"]
        # Criterion
        if mode == 'l2':
            self.criterion = nn.MSELoss()
        elif mode == 'l1':
            self.criterion = nn.L1Loss()
        else:
            raise NotImplementedError()

    def __call__(self, predictions, ref_imgs):

        preds_position = predictions[:, :, :2]
        preds_color = predictions[:, :, 5:]

        ref_imgs = self.blur(ref_imgs)
        ref_imgs = repeat(ref_imgs, 'bs ch h w -> (bs L) ch h w', L=predictions.size(1))
        grid = rearrange(preds_position, 'bs L dim -> (bs L) 1 1 dim')
        if self.detach_grid:
            grid = grid.detach()
        target_color_img = F.grid_sample(ref_imgs, 2 * grid - 1, align_corners=False, padding_mode='border')
        target_color_img = rearrange(target_color_img, '(bs L) ch 1 1 -> bs L ch', L=predictions.size(1))
        loss = self.criterion(preds_color, target_color_img)

        return loss


class RenderImageLoss(nn.Module):
    def __init__(self, config):
        super(RenderImageLoss, self).__init__()

        self.type = config["train"]["losses"]["reference_img"]["render"]["type"]
        self.active = config["train"]["losses"]["reference_img"]["render"]["weight"] > 0
        mode = config["train"]["losses"]["reference_img"]["render"]["mode"]
        meta_brushes_path = config["renderer"]["brushes_path"]
        canvas_size = config["dataset"]["resize"]

        if self.type == 'full':
            # render the strokes
            self.renderer = LightRenderer(brushes_path=meta_brushes_path, canvas_size=canvas_size)
            # reduction = 'none'
            reduction = 'mean'
        elif self.type == 'masked':
            # render the strokes
            self.renderer = LightRenderer(brushes_path=meta_brushes_path, canvas_size=canvas_size)
            # reduction = 'none'
            reduction = 'none'
        else:
            raise NotImplementedError()

        # Criterion
        if mode == 'l2':
            self.criterion = nn.MSELoss(reduction=reduction)
        elif mode == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        else:
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


# ======================================================================================================================
# DistReg
class DistributionRegularization(nn.Module):
    def __init__(self, config):
        super(DistributionRegularization, self).__init__()
        tmp_config = config["train"]["losses"]["regularization_dist"]

        self.active = tmp_config["weight"] > 0
        self.K = tmp_config["K"]  # Number of neighbors to compute similarity
        self.param_per_stroke = config["model"]["n_strokes_params"]

        self.mode = tmp_config["mode"]

        self.L = config["dataset"]["context_length"] + config["dataset"]["sequence_length"]
        self.get_index(L=self.L, K=self.K)

        self.reweight_kl = tmp_config["reweight_kl"]
        if self.reweight_kl:
            # Remove (h,w,theta) from parameters before computing the features
            self.param_per_stroke -= 3

        self.dim_features = self.n * self.param_per_stroke

    def get_index(self, L, K=10):
        ids = []
        for i in range(L):
            for j in range(L):
                if (j > i + K) or (j < i - K) or i == j:
                    continue
                else:
                    ids.append([i, j])
        ids = np.array(ids)
        self.id0 = ids[:, 0]
        self.id1 = ids[:, 1]
        self.n = ids.shape[0]

    def compute_features(self, x):
        bs = x.shape[0]
        feat = torch.empty((bs, self.dim_features))
        if self.reweight_kl:
            # TODO: fix here, we remove (w,h, theta)
            param_list = [0, 1, 5, 6, 7]
        else:
            param_list = [0, 1, 2, 3, 4, 5, 6, 7]

        for j in range(len(param_list)):
            feat[:, j * self.n: (j + 1) * self.n] = x[:, self.id0, param_list[j]] - x[:, self.id1, param_list[j]]
        return feat.t().contiguous()

    def fast_fid(self, x, y, eps: float = 1e-6) -> float:
        # https://arxiv.org/pdf/2009.14075.pdf

        _, m = x.shape
        mu_x = torch.mean(x, dim=1, keepdim=False)
        C_x = (x - mu_x[:, None]) / np.sqrt(m - 1)
        sigma_x = C_x.mm(C_x.t())

        # Load the pre-computed mu and sigma if exists
        if self.precomputed_dataset_statistics:
            mu_y = self.mu_dataset.to(x.device)
            sigma_y = self.sigma_dataset.to(x.device)
        else:
            mu_y = torch.mean(y, dim=1, keepdim=False)
            C_y = (y - mu_y[:, None]) / np.sqrt(m - 1)
            sigma_y = C_y.mm(C_y.t())

        # Compute the FID
        diff = mu_x - mu_y

        # Compute S
        S = C_x.t().mm(sigma_y).mm(C_x)
        e, _ = torch.linalg.eig(S)
        tr_covmean = torch.sum(torch.abs(torch.sqrt(e)))

        return diff.dot(diff) + torch.trace(sigma_x) + torch.trace(sigma_y) - 2 * tr_covmean

    def kl(self, params, reference_params, eps=1e-8):

        variance, mean = torch.var_mean(params, dim=-1)
        log_variance = torch.log(variance)

        reference_variance, reference_mean = torch.var_mean(reference_params, dim=-1)
        reference_log_variance = torch.log(reference_variance)

        variance = torch.clamp(variance, min=eps)
        reference_variance = torch.clamp(reference_variance, min=eps)

        variance_ratio = variance / reference_variance
        mus_term = (reference_mean - mean).pow(2) / reference_variance
        kl = reference_log_variance - log_variance - 1 + variance_ratio + mus_term

        kl = kl.sum(dim=-1)
        kl = 0.5 * kl.mean()
        return kl

    def __call__(self, preds, batch):

        if not self.active:
            return torch.tensor([0], device=preds.device)

        # Original
        x_real = torch.cat((batch['strokes_ctx'], batch['strokes_seq']), dim=1)  # cat on length dim
        f_real = self.compute_features(x_real)

        # Predictions
        x_pred = torch.cat((batch['strokes_ctx'], preds), dim=1)
        # Compute features
        f_pred = self.compute_features(x_pred)

        # Compute divergence between distributions
        if self.mode == 'fid':
            loss = self.fast_fid(f_pred, f_real)
        elif self.mode == 'kl':
            loss = self.kl(params=f_pred, reference_params=f_real)

        return loss
