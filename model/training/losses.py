import torch
import torch.nn as nn


class KLDivergence(nn.Module):

    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, mu, log_sigma):
        kl_loss = 1 + log_sigma - mu.pow(2) - log_sigma.exp()   # bs x n_features
        kl_loss = -0.5 * torch.sum(kl_loss, dim=-1)

        return torch.mean(kl_loss)


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, pred, labels):
        loss = (pred - labels).pow(2).sum(dim=[1, 2])   # sum over features and sequence length
        return loss.mean()