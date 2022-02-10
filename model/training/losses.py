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
    def __init__(self, config):
        super(ColorImageLoss, self).__init__()
        self.blur = torchvision.transforms.GaussianBlur(kernel_size=(7, 7))
        mode = config["train"]["losses"]["reference_img"]["color"]["mode"]
        self.detach_grid = config["train"]["losses"]["reference_img"]["color"]["detach"]
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
        grid = rearrange(preds_position, 'bs L dim -> (bs L) 1 1 dim')
        if self.detach_grid:
            grid = grid.detach()
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
    def __init__(self, config) :
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

        mode = config["train"]["losses"]["reference_img"]["pos_color"]["mode"]
        self.active = config["train"]["losses"]["reference_img"]["pos_color"]["weight"] > 0
        self.K = config["train"]["losses"]["reference_img"]["pos_color"]["K"]

        # Criterion
        if mode == 'l2' :
            self.p = 2
        elif mode == 'l1' :
            self.p = 1
        else :
            raise NotImplementedError()

    def __call__(self, predictions, ref_imgs) :

        if not self.active :
            return torch.tensor([0], device=predictions.device)

        bs, L, _ = predictions.shape
        img_size = ref_imgs.shape[-1]

        preds_position = predictions[:, :, :2]

        # Sample GT colors
        feat_temp = repeat(ref_imgs, 'bs ch h w -> (L bs) ch h w', L=L)
        grid = rearrange(preds_position, 'L bs p -> (L bs) 1 1 p')
        pooled_colors = F.grid_sample(feat_temp, 2 * grid - 1, align_corners=False, mode='nearest')

        feat_temp = rearrange(feat_temp, '(L bs) ch h w -> bs L ch h w', bs=bs, L=L)
        pooled_colors = rearrange(pooled_colors, '(L bs) ch 1 1 -> bs L ch 1 1', bs=bs, L=L)
        pooled_colors = repeat(pooled_colors, 'bs L ch 1 1 -> bs L ch H W', H=img_size, W=img_size)

        # Find top K similar color across the image
        similar_colors = F.mse_loss(feat_temp, pooled_colors, reduction='none').sum(dim=2)
        _, idx = similar_colors.reshape(bs, L, -1).topk(k=self.K, largest=False)

        # Convert idx to (x,y)
        tgt_x = torch.remainder(idx, img_size) / img_size
        tgt_y = torch.div(idx, img_size, rounding_mode='floor') / img_size

        tgt = torch.stack([tgt_x.reshape(bs, L * self.K), tgt_y.reshape(bs, L * self.K)], dim=-1)
        tgt = tgt.reshape(bs, L, self.K, 2)

        # Shift the target up and down by 1 position, i.e. compare each element of the sequence with the precedent
        # and the following one
        tgt_down = torch.roll(tgt, shifts=1, dims=1)
        pos_detached = repeat(preds_position, 'bs L p -> bs L K p', K=self.K).detach()

        # Computes distances
        dist_down = F.mse_loss(pos_detached, tgt_down, reduction='none').sum(dim=-1)
        _, closest_tgt_down = torch.min(dist_down, dim=-1)

        #
        final_tgt = tgt_down.reshape(bs * L, self.K, 2)[torch.arange(bs * L), closest_tgt_down.view(bs * L)]
        final_tgt = final_tgt.reshape(bs, L, 2)

        # compute L2
        loss = F.mse_loss(preds_position[:, 1:], final_tgt[:, 1:], reduction='none').sum(dim=-1)

        return torch.mean(loss)

############################################################################################################

class CCLoss(nn.Module) :
    def __init__(self, config) :
        '''
        Use all the kNN
        '''
        super(CCLoss, self).__init__()

        mode = config["train"]["losses"]["reference_img"]["pos_color"]["mode"]
        self.active = config["train"]["losses"]["reference_img"]["pos_color"]["weight"] > 0
        self.find_target = config["train"]["losses"]["reference_img"]["pos_color"]["find_target"] # wa = weighted average, knn
        self.tau = config["train"]["losses"]["reference_img"]["pos_color"]["tau"]
        self.p = config["train"]["losses"]["reference_img"]["pos_color"]["p"]
        self.K = config["train"]["losses"]["reference_img"]["pos_color"]["K"]

        # Criterion
        if mode == 'l2' :
            self.p = 2
        elif mode == 'l1' :
            self.p = 1
        else :
            raise NotImplementedError()


    def find_target_knn(self, color_similarity, preds_position):
        bs, L, img_size, _ = color_similarity.shape
        _, idx = color_similarity.reshape(bs, L, -1).topk(k=self.K, largest=False)

        # Convert idx to (x,y)
        tgt_x = torch.remainder(idx, img_size) / img_size
        tgt_y = torch.div(idx, img_size, rounding_mode='floor') / img_size

        tgt = torch.stack([tgt_x.reshape(bs, L * self.K), tgt_y.reshape(bs, L * self.K)], dim=-1)
        tgt = tgt.reshape(bs, L, self.K, 2)

        # Shift the target up and down by 1 position, i.e. compare each element of the sequence with the precedent
        # and the following one
        tgt_down = torch.roll(tgt, shifts=1, dims=1)
        pos_detached = repeat(preds_position, 'bs L p -> bs L K p', K=self.K).detach()

        # Computes distances
        dist_down = F.mse_loss(pos_detached, tgt_down, reduction='none').sum(dim=-1)
        _, closest_tgt_down = torch.min(dist_down, dim=-1)

        #
        final_tgt = tgt_down.reshape(bs * L, self.K, 2)[torch.arange(bs * L), closest_tgt_down.view(bs * L)]
        final_tgt = final_tgt.reshape(bs, L, 2)

        return final_tgt

    def find_target_wa(self, color_similarity, preds_position):
        bs, L, img_size, _ = color_similarity.shape

        b_id, seq_id, y_id, x_id = torch.where(color_similarity < self.tau)
        tgt = torch.stack((x_id, y_id), dim=-1) / img_size

        # Compute final Targets
        final_tgt = torch.zeros(bs, L, 2, device=preds_position.device)
        for b in range(bs) :
            for t_minus_one in range(L - 1) :
                iids = torch.logical_and(b_id == b, seq_id == t_minus_one)
                nn_t_minus_one = tgt[iids]
                # Distance between NN at time T-1, and predictions at time T
                dist = torch.cdist(preds_position[b, t_minus_one + 1][None], nn_t_minus_one, p=2) + torch.tensor(1e-9, device=preds_position.device)
                w = torch.pow(1 / dist, self.p)  # inverse of the distance
                w_norm = w / w.sum()

                # Target at time T is the weighted average of NN at time T-1
                final_tgt[b, t_minus_one + 1, :] = torch.sum(w_norm.unsqueeze(-1) * nn_t_minus_one, dim=1)

        return final_tgt

    def __call__(self, predictions, ref_imgs) :
        if not self.active :
            return torch.tensor([0], device=predictions.device)

        bs, L, _ = predictions.shape
        img_size = ref_imgs.shape[-1]

        preds_position = predictions[:, :, :2]

        # Sample GT colors
        feat_temp = repeat(ref_imgs, 'bs ch h w -> (L bs) ch h w', L=L)
        grid = rearrange(preds_position, 'L bs p -> (L bs) 1 1 p')
        pooled_colors = F.grid_sample(feat_temp, 2 * grid - 1, align_corners=False, mode='nearest')

        feat_temp = rearrange(feat_temp, '(L bs) ch h w -> bs L ch h w', bs=bs, L=L)
        pooled_colors = rearrange(pooled_colors, '(L bs) ch 1 1 -> bs L ch 1 1', bs=bs, L=L)
        pooled_colors = repeat(pooled_colors, 'bs L ch 1 1 -> bs L ch H W', H=img_size, W=img_size)

        # Find similar color across the image
        similar_colors = F.mse_loss(feat_temp, pooled_colors, reduction='none').sum(dim=2)

        # Compute the target position
        if self.find_target == "wa":
            target = self.find_target_wa(color_similarity=similar_colors, preds_position=preds_position.detach())
        elif self.find_target == "knn":
            target = self.find_target_knn(color_similarity=similar_colors, preds_position=preds_position.detach())
        else:
            raise NotImplementedError()

        # compute L2
        loss = F.mse_loss(preds_position[:, 1 :], target[:, 1 :], reduction='none').sum(dim=-1)
        return torch.mean(loss)


#########################################################################################################
from torch.autograd import Function
import scipy
import numpy as np


class MatrixSquareRoot(Function) :
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """

    @staticmethod
    def forward(ctx, input) :
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output) :
        grad_input = None
        if ctx.needs_input_grad[0] :
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply

class FIDLoss(nn.Module) :
    def __init__(self, config) :
        super(FIDLoss, self).__init__()

        self.active = config["train"]["losses"]["fid"]["weight"] > 0
        self.K = config["train"]["losses"]["fid"]["K"] # Number of neighbors to compute similarity
        self.param_per_stroke = config["model"]["n_strokes_params"]

        #L = config["dataset"]["context_length"] + config["dataset"]["sequence_length"]
        L = config["dataset"]["sequence_length"]
        ids = []
        for i in range(L) :
            for j in range(L) :
                if (j > i + self.K) or (j < i - self.K) or i == j :
                    continue
                else :
                    ids.append([i, j])

        ids = torch.tensor(ids)
        self.id0 = ids[:, 0]
        self.id1 = ids[:, 1]
        self.n = ids.shape[0]

        self.dim_features = self.n * self.param_per_stroke

    def compute_features(self, x) :
        bs = x.shape[0]
        feat = torch.empty((bs, self.dim_features))
        for j in range(self.param_per_stroke) :
            feat[:, j * self.n : (j + 1) * self.n] = x[:, self.id0, j] - x[:, self.id1, j]
        return feat.t().contiguous()

    def fid_score_torch(self,
                        mu1: torch.Tensor,
                        mu2: torch.Tensor,
                        sigma1: torch.Tensor,
                        sigma2: torch.Tensor,
                        eps: float = 1e-6) -> float :
        try :
            import numpy as np
        except ImportError :
            raise RuntimeError("fid_score requires numpy to be installed.")

        try :
            import scipy.linalg
        except ImportError :
            raise RuntimeError("fid_score requires scipy to be installed.")

        # mu1, mu2 = mu1.cpu(), mu2.cpu()
        # sigma1, sigma2 = sigma1.cpu(), sigma2.cpu()

        diff = mu1 - mu2

        # Product might be almost singular
        offset = torch.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1+offset).mm(sigma2+offset))
        # Numerical error might give slight imaginary component

        if torch.is_complex(covmean) :
            if not torch.allclose(torch.diagonal(covmean), torch.tensor([0.0], dtype=torch.double), atol=1e-3) :
                m = torch.max(torch.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = torch.trace(covmean)

        if not torch.isfinite(covmean).all() :
            tr_covmean = torch.sum(torch.sqrt(((torch.diag(sigma1) * eps) * (torch.diag(sigma2) * eps)) / (eps * eps)))

        return diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean


    def __call__(self, ctx, seq, preds):

        if not self.active:
            return torch.tensor([0], device=preds.device)
        # Real sequence
        # real_seq = torch.cat((ctx, seq), dim=1)
        real_seq = self.compute_features(seq)

        real_cov = torch.cov(real_seq)
        real_mean = torch.mean(real_seq, dim=-1)

        # Generated seq
        #gen_seq = torch.cat((ctx, preds), dim=1)
        gen_seq = self.compute_features(preds)
        gen_cov = torch.cov(gen_seq)
        gen_mean = torch.mean(gen_seq, dim=-1)

        return self.fid_score_torch(mu1=real_mean, mu2=gen_mean, sigma1=real_cov, sigma2=gen_cov)