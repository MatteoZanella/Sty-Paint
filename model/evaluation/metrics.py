import numpy as np
from scipy import linalg
import torch


# Code from https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py

class FDMetric :
    def __init__(self, n_files, seq_len=8) :

        self.param_per_stroke = 11   # leave out the last parameter, i.e. alpha which is not used
        self.n_files = n_files
        self.ids = np.array([i for i in itertools.permutations(range(seq_len), 2)])
        self.n = self.ids.shape[0]

        #print(f'Numebr of permutations : {self.n}')

        self.dim_features = self.n * self.param_per_stroke
        self.original_features = np.empty((self.n_files, self.dim_features))
        self.generated_features = np.empty((self.n_files, self.dim_features))
        self.counter = 0

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

    def update_queue(self, original, generated):

        # Original
        self.original_features[self.counter, :] = self.compute_features(original)
        # Generated
        self.generated_features[self.counter, :] = self.compute_features(generated)
        # Update
        self.counter += 1

    def compute_features(self, x) :
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        feat = np.empty(self.dim_features)
        for j in range(self.param_per_stroke):
            feat[j * self.n : (j + 1) * self.n] = x[:, self.ids[:, 0], j] - x[:, self.ids[:, 1], j]
        return feat

    def compute_mean_cov(self):
        generated = dict(mu_all=np.mean(self.generated_features, axis=0),
                   cov_all=np.cov(self.generated_features, rowvar=False),
                   mu_position=np.mean(self.generated_features[:, :6 * self.n]),
                   cov_position=np.cov(self.generated_features[:, :6 * self.n], rowvar=False),
                   mu_color=np.mean(self.generated_features[:, 6 * self.n]),
                   cov_color=np.cov(self.generated_features[:, 6 * self.n], rowvar=False))

        original =  dict(mu_all=np.mean(self.original_features, axis=0),
                   cov_all=np.cov(self.original_features, rowvar=False),
                   mu_position=np.mean(self.original_features[:, :6 * self.n]),
                   cov_position=np.cov(self.original_features[:, :6 * self.n], rowvar=False),
                   mu_color=np.mean(self.original_features[:, 6 * self.n]),
                   cov_color=np.cov(self.original_features[:, 6 * self.n], rowvar=False))

        return generated, original

    def compute_fd(self):
        assert self.counter == self.n_files
        gen, orig = self.compute_mean_cov()

        output = dict(
            all = self._calculate_frechet_distance(mu1=gen['mu_all'], sigma1=gen['cov_all'], mu2=orig['mu_all'], sigma2=orig['cov_all']),
            position = self._calculate_frechet_distance(mu1=gen['mu_position'], sigma1=gen['cov_position'], mu2=orig['mu_position'], sigma2=orig['cov_position']),
            color = self._calculate_frechet_distance(mu1=gen['mu_color'], sigma1=gen['cov_color'], mu2=orig['mu_color'], sigma2=orig['cov_color']))

        return output


# ======================================================================================================================
import lpips
import itertools
import torchvision.transforms as transforms


# code from https://github.com/richzhang/PerceptualSimilarity, pip install lpips
class LPIPSMetric :
    def __init__(self, fn_backbone='vgg') :
        self.loss_fn = lpips.LPIPS(net=fn_backbone)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # def prepare_input(self, x):
    #     bs = x.size(0)
    #     x = rearrange(x, 'bs L c h w -> (bs L) c h w')
    #     x = 2 * x - 1
    #     x = rearrange(x, '(bs L) c h w -> bs L c h w', bs = bs)
    #     return x

    def get_combinations(self, n) :
        return np.array([i for i in itertools.combinations(range(n), 2)])

    @torch.no_grad()
    def __call__(self, x) :
        """
        Args:
            x: tensor of size [n_samples x H x W x 3], contains n_samples continuations suggested by the model
        Returns:
        """
        n_samples = len(x)
        idxs = self.get_combinations(n_samples)

        x_norm = [self.transform(x_i) for x_i in x]
        x_norm = torch.stack(x_norm)

        loss = []
        for idx in idxs :
            loss.append(self.loss_fn(x_norm[idx[0]],
                                     x_norm[idx[1]]))

        loss = torch.tensor(loss)
        return loss.mean()


    def cleanLPIPS(self, ref, gen):
        gen = self.transform(gen)
        return self.loss_fn(ref, gen)


# ======================================================================================================================
from einops import repeat
import torch.nn.functional as F


class MaskedL2 :
    def __init__(self) :
        pass

    def normalize_0_1(self, x) :
        # Images are normalized between -1, 1. Shift to 0, 1
        x = 0.5 * (x + 1)
        return x[0].permute(1, 2, 0).cpu().numpy()

    @torch.no_grad()
    def __call__(self, ref, gen, alphas) :
        """
        Args:
            ref: reference images
            gen: render of each stroke
            alphas: alpha matrix of the last strokes

        Returns:

        """
        ref = self.normalize_0_1(ref)
        alphas = (np.stack(alphas).sum(axis=0) > 0)[:, :, None]
        loss_masked = (((ref - gen) * alphas) ** 2).sum()
        loss_normalized = loss_masked / alphas.sum()

        return loss_masked, loss_normalized
