import numpy as np
from scipy import linalg
import torch
from einops import rearrange

def compute_color_difference(x):
    x = x[:, :, 5:]
    if torch.is_tensor(x):
        l1 = torch.abs(torch.diff(x, dim=1)).mean()
        l2 = torch.pow(torch.diff(x, dim=1), 2).mean()
    else:
        l1 = np.abs(np.diff(x, axis=1)).sum(axis=1).mean()
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
class FDMetric :
    def __init__(self, seq_len=8) :
        """
        Compute batched FD metric
        """

        self.param_per_stroke = 8
        ids = np.array([i for i in itertools.permutations(range(seq_len), 2)])
        self.id0 = ids[:, 0]
        self.id1 = ids[:, 1]
        self.n = ids.shape[0]

        #print(f'Numebr of permutations : {self.n}')

        self.dim_features = self.n * self.param_per_stroke
        self.keys = ['all', 'position', 'color']   # Divide the FD for these parameters

    def _calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6) :
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """

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

    def compute_features(self, x) :
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        bs = x.shape[0]
        feat = np.empty((bs, self.dim_features))
        for j in range(self.param_per_stroke):
            feat[:, j * self.n : (j + 1) * self.n] = x[:, self.id0, j] - x[:, self.id1, j]
        return feat

    def compute_mean_cov(self, feat):
        out = dict(mu_all=np.mean(feat, axis=0),
                   cov_all=np.cov(feat, rowvar=False),
                   mu_position=np.mean(feat[:, :5 * self.n]),
                   cov_position=np.cov(feat[:, :5 * self.n], rowvar=False),
                   mu_color=np.mean(feat[:, 5 * self.n:]),
                   cov_color=np.cov(feat[:, 5 * self.n:], rowvar=False))
        return out

    def __call__(self, original, generated):
        # Original
        orig_feat = self.compute_features(original)
        orig = self.compute_mean_cov(orig_feat)

        # Generated
        gen_feat = self.compute_features(generated)
        gen = self.compute_mean_cov(gen_feat)

        output = dict(
            all = self._calculate_frechet_distance(mu1=gen['mu_all'], sigma1=gen['cov_all'], mu2=orig['mu_all'], sigma2=orig['cov_all']),
            position = self._calculate_frechet_distance(mu1=gen['mu_position'], sigma1=gen['cov_position'], mu2=orig['mu_position'], sigma2=orig['cov_position']),
            color = self._calculate_frechet_distance(mu1=gen['mu_color'], sigma1=gen['cov_color'], mu2=orig['mu_color'], sigma2=orig['cov_color']))

        return output

########################################################################################################################
class FDMetricIncremental :
    def __init__(self, seq_len=8) :
        """
        Compute batched FD metric
        """

        self.param_per_stroke = 8
        ids = np.array([i for i in itertools.permutations(range(seq_len), 2)])
        self.id0 = ids[:, 0]
        self.id1 = ids[:, 1]
        self.n = ids.shape[0]

        #print(f'Numebr of permutations : {self.n}')

        self.dim_features = self.n * self.param_per_stroke
        self.original_features = []
        self.generated_features = []
        self.keys = ['all', 'position', 'color']   # Divide the FD for these parameters

    def _calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6) :
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """

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

    def compute_features(self, x) :
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        bs = x.shape[0]
        feat = np.empty((bs, self.dim_features))
        for j in range(self.param_per_stroke):
            feat[:, j * self.n : (j + 1) * self.n] = x[:, self.id0, j] - x[:, self.id1, j]
        return feat

    def compute_mean_cov(self):
        self.generated_features = np.concatenate(self.generated_features)
        generated = dict(mu_all=np.mean(self.generated_features, axis=0),
                   cov_all=np.cov(self.generated_features, rowvar=False),
                   mu_position=np.mean(self.generated_features[:, :5 * self.n]),
                   cov_position=np.cov(self.generated_features[:, :5 * self.n], rowvar=False),
                   mu_color=np.mean(self.generated_features[:, 5 * self.n:]),
                   cov_color=np.cov(self.generated_features[:, 5 * self.n:], rowvar=False))


        # Original
        self.original_features = np.concatenate(self.original_features)
        original =  dict(mu_all=np.mean(self.original_features, axis=0),
                   cov_all=np.cov(self.original_features, rowvar=False),
                   mu_position=np.mean(self.original_features[:, :5 * self.n]),
                   cov_position=np.cov(self.original_features[:, :5 * self.n], rowvar=False),
                   mu_color=np.mean(self.original_features[:, 5 * self.n:]),
                   cov_color=np.cov(self.original_features[:, 5 * self.n:], rowvar=False))

        return generated, original

    def update_queue(self, original, generated):
        original = original[:, :, :self.param_per_stroke]
        generated = generated[:, :, :self.param_per_stroke]

        self.original_features.append(self.compute_features(original))
        self.generated_features.append(self.compute_features(generated))

    def compute_fd(self):
        gen, orig = self.compute_mean_cov()

        output = dict(
            all = self._calculate_frechet_distance(mu1=gen['mu_all'], sigma1=gen['cov_all'], mu2=orig['mu_all'], sigma2=orig['cov_all']),
            position = self._calculate_frechet_distance(mu1=gen['mu_position'], sigma1=gen['cov_position'], mu2=orig['mu_position'], sigma2=orig['cov_position']),
            color = self._calculate_frechet_distance(mu1=gen['mu_color'], sigma1=gen['cov_color'], mu2=orig['mu_color'], sigma2=orig['cov_color']))

        return output

# ======================================================================================================================
import lpips
import itertools


# code from https://github.com/richzhang/PerceptualSimilarity, pip install lpips
class LPIPSDiversityMetric:
    def __init__(self, fn_backbone='vgg') :
        self.loss_fn = lpips.LPIPS(net=fn_backbone)

    def get_combinations(self, n) :
        return np.array([i for i in itertools.combinations(range(n), 2)])

    @torch.no_grad()
    def __call__(self, x) :
        """
        Args:
            x: numpy array of size [bs x n_samples x H x W x 3], contains n_samples continuations suggested by the model
        Returns:
        """
        bs = x.shape[0]
        n_samples = x.shape[1]
        idxs = self.get_combinations(n_samples)

        x_tensor = torch.empty(x.shape)
        for b in range(bs):
            for l in range(n_samples):
                x_tensor[b,l] = torch.tensor( 0.5 * x[b, l] + 1)   # shift images in the range [-1,1]

        x_tensor = x_tensor.permute(0, 1, 4, 2, 3)
        loss = []
        for b in range(bs):
            for idx in idxs:
                loss.append(self.loss_fn(x_tensor[b, idx[0]],
                                         x_tensor[b, idx[1]]))

        loss = torch.tensor(loss)
        return loss.mean()



class FeaturesDiversity:
    def __init__(self) :
        seq_len = 8
        self.param_per_stroke = 8
        ids = np.array([i for i in itertools.permutations(range(seq_len), 2)])
        self.id0 = ids[:, 0]
        self.id1 = ids[:, 1]
        self.n = ids.shape[0]

        # print(f'Numebr of permutations : {self.n}')

        self.dim_features = self.n * self.param_per_stroke

    def get_combinations(self, n) :
        return np.array([i for i in itertools.combinations(range(n), 2)])

    def compute_features(self, x) :
        if torch.is_tensor(x) :
            x = x.detach().cpu().numpy()
        bs = x.shape[0]
        n_samples = x.shape[1]

        feat = np.empty((bs, n_samples, self.dim_features))
        for j in range(self.param_per_stroke) :
            for n in range(n_samples) :
                feat[:, n, j * self.n : (j + 1) * self.n] = x[:, n, self.id0, j] - x[:, n, self.id1, j]
        return feat

    def mse(self, x, y) :
        return np.square(np.subtract(x, y)).mean()

    def __call__(self, x) :
        """
        Args:
            x: [bs x n_samples x len x params]
        Returns:
        """
        bs = x.shape[0]
        n_samples = x.shape[1]
        idxs = self.get_combinations(n_samples)

        x_feat = self.compute_features(x)

        loss = []
        for b in range(bs) :
            for idx in idxs :
                loss.append(self.mse(x_feat[b, idx[0]], x_feat[b, idx[1]]))
        loss = np.array(loss)
        return loss.mean()



# ======================================================================================================================
from einops import repeat
from tslearn.metrics import dtw

def maskedL2(ref_imgs, frames, alphas) :
    """
    All input between 0, 1 range
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
    loss = loss.sum(axis=(2,3,4)) / area

    return loss.mean()

def compute_dtw(x, y) :
    bs = x.shape[0]
    dtw_scores = np.empty(bs)
    for b in range(bs) :
        dtw_scores[b] = dtw(x[b], y[b])
    return dtw_scores.mean()
