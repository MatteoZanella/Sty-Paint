import numpy as np
from scipy import linalg
import torch
from einops import repeat
from tslearn.metrics import dtw
import lpips
import itertools
from . import tools


def compute_color_difference(x):
    x = x[:, :, 5:]
    if torch.is_tensor(x):
        l1 = torch.abs(torch.diff(x, dim=1)).mean()
        l2 = torch.pow(torch.diff(x, dim=1), 2).mean()
    else:
        l1 = np.abs(np.diff(x, axis=1)).mean()
        l2 = np.square(np.diff(x, axis=1)).mean()

    return l1, l2

class WassersteinDistance:
    def __init__(self):
        pass

    def get_sigma_sqrt(self, w, h, theta) :
        sigma_00 = w * (torch.cos(theta) ** 2) / 2 + h * (torch.sin(theta) ** 2) / 2
        sigma_01 = (w - h) * torch.cos(theta) * torch.sin(theta) / 2
        sigma_11 = h * (torch.cos(theta) ** 2) / 2 + w * (torch.sin(theta) ** 2) / 2
        sigma_0 = torch.stack([sigma_00, sigma_01], dim=-1)
        sigma_1 = torch.stack([sigma_01, sigma_11], dim=-1)
        sigma = torch.stack([sigma_0, sigma_1], dim=-2)
        return sigma

    def get_sigma(self, w, h, theta) :
        sigma_00 = w * w * (torch.cos(theta) ** 2) / 4 + h * h * (torch.sin(theta) ** 2) / 4
        sigma_01 = (w * w - h * h) * torch.cos(theta) * torch.sin(theta) / 4
        sigma_11 = h * h * (torch.cos(theta) ** 2) / 4 + w * w * (torch.sin(theta) ** 2) / 4
        sigma_0 = torch.stack([sigma_00, sigma_01], dim=-1)
        sigma_1 = torch.stack([sigma_01, sigma_11], dim=-1)
        sigma = torch.stack([sigma_0, sigma_1], dim=-2)
        return sigma

    def gaussian_w_distance(self, param_1, param_2) :
        """
        Args:
            param_1: bs x length x 5
            param_2:
        Returns:
            should average all to have a fair result
        """
        mu_1, w_1, h_1, theta_1 = torch.split(param_1, (2, 1, 1, 1), dim=-1)
        w_1 = w_1.squeeze(-1)
        h_1 = h_1.squeeze(-1)
        theta_1 = torch.acos(torch.tensor(-1., device=param_1.device)) * theta_1.squeeze(-1)
        trace_1 = (w_1 ** 2 + h_1 ** 2) / 4
        mu_2, w_2, h_2, theta_2 = torch.split(param_2, (2, 1, 1, 1), dim=-1)
        w_2 = w_2.squeeze(-1)
        h_2 = h_2.squeeze(-1)
        theta_2 = torch.acos(torch.tensor(-1., device=param_2.device)) * theta_2.squeeze(-1)
        trace_2 = (w_2 ** 2 + h_2 ** 2) / 4
        sigma_1_sqrt = self.get_sigma_sqrt(w_1, h_1, theta_1)
        sigma_2 = self.get_sigma(w_2, h_2, theta_2)
        trace_12 = torch.matmul(torch.matmul(sigma_1_sqrt, sigma_2), sigma_1_sqrt)
        trace_12 = torch.sqrt(trace_12[..., 0, 0] + trace_12[..., 1, 1] + 2 * torch.sqrt(
            trace_12[..., 0, 0] * trace_12[..., 1, 1] - trace_12[..., 0, 1] * trace_12[..., 1, 0]))
        return torch.sum((mu_1 - mu_2) ** 2, dim=-1) + trace_1 + trace_2 - 2 * trace_12

    def __call__(self, param1, param2):
        assert param1.shape[-1] == 5
        assert param1.shape == param2.shape

        loss = self.gaussian_w_distance(param1, param2)

        return loss.mean()  # average over the batch/length dimension

#######################################################################################################################
# Frechet Strokes Distance
class FSD:
    def __init__(self):
        super(FSD, self).__init__()

    def __call__(self, generated, original, ctx=None):

        generated = tools.compute_features(generated, ctx=ctx)
        original = tools.compute_features(original, ctx=ctx)

        fsd = self.calculate_frechet_distance(generated["feat"], original["feat"])
        fsd_pos = self.calculate_frechet_distance(generated["feat_pos"], original["feat_pos"])
        fsd_color = self.calculate_frechet_distance(generated["feat_color"], original["feat_color"])

        return dict(fsd=fsd, fsd_pos=fsd_pos, fsd_color=fsd_color)

    def calculate_frechet_distance(self, x, y, compute_mean_cov=True, eps=1e-6):

        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()

        if compute_mean_cov:
            mu1 = np.mean(x, axis=0)
            sigma1 = np.cov(x, rowvar=False)
            mu2 = np.mean(y, axis=0)
            sigma2 = np.cov(y, rowvar=False)

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all() :
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean) :
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3) :
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)
# ======================================================================================================================
# LPIPS Diversity
# code from https://github.com/richzhang/PerceptualSimilarity, pip install lpips
class LPIPSDiversityMetric:
    def __init__(self, fn_backbone='vgg') :
        self.loss_fn = lpips.LPIPS(net=fn_backbone)

    def get_combinations(self, n) :
        return np.array([i for i in itertools.combinations(range(n), 2)])

    def rearrange_visuals(self, frames, alpha):

        bs, n_samples, L, H, W, c = frames.shape
        output = torch.empty((bs, n_samples, H, W, c))
        for b in range(bs) :
            for n in range(n_samples) :
                rec = frames[b, n, 0]
                for ii in range(1, L) :
                    rec = frames[b, n, ii] * alpha[b, n, ii] + rec * (1 - alpha[b, n, ii])
                output[b, n] = 2 * torch.tensor(rec) - 1   # shift to [-1, 1]

        return output

    @torch.no_grad()
    def __call__(self, x, a) :
        """
        Args:
            x: numpy array of size [bs x n_samples x H x W x 3], contains n_samples continuations suggested by the model
        Returns:
        """
        bs = x.shape[0]
        n_samples = x.shape[1]
        idxs = self.get_combinations(n_samples)

        x_tensor = self.rearrange_visuals(x, a)
        x_tensor = x_tensor.permute(0, 1, 4, 2, 3)
        loss = []
        for b in range(bs):
            for idx in idxs:
                loss.append(self.loss_fn(x_tensor[b, idx[0]],
                                         x_tensor[b, idx[1]]))

        loss = torch.tensor(loss)
        return loss.mean()

# ======================================================================================================================
# Stroke Color L2
class StrokeColorL2:
    def __init__(self):
        super(StrokeColorL2, self).__init__()

    def __call__(self, ref_imgs, frames, alphas):
        """
        All input range [0,1].
        Args:
            ref: [bs x 3 x h x w]
            frames: [bs x L x h x w x 3]  rendered frames
            alphas: [bs x L x h x w x 1]  0, 1 binary map showing the transparency matrix

        Returns:

        """
        L = frames.shape[1]

        ref_imgs = repeat(ref_imgs.cpu(), 'bs ch h w -> bs L h w ch', L=L)
        ref_imgs = ref_imgs.numpy()

        loss = np.square(np.subtract(ref_imgs, frames) * alphas)
        area = alphas.sum(axis=(2, 3, 4))
        loss = loss.sum(axis=(2, 3, 4)) / area

        return loss.mean()

# ======================================================================================================================
# DTW
class DTW:
    def __init__(self):
        super(DTW, self).__init__()

    def __call__(self, x, y):
        bs = x.shape[0]
        output = np.empty(bs)
        for b in range(bs):
            output[b] = dtw(x[b], y[b])
        return output.mean()
