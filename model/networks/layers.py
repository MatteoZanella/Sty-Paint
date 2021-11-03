import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange, repeat
import torch.nn.functional as F


########################################################################################################################
class PEWrapper :
    def __init__(self, config) :
        self.input_dim = config["dataset"]["resize"]
        self.encoder_dim = config["model"]["img_encoder"]["visual_feat_hw"]
        self.d_model = config["model"]["d_model"]

    def pe_visual_tokens(self, device):
        pe_spatial, pe_time = self.positionalencoding3d(self.encoder_dim, self.encoder_dim, 1, self.d_model, device)
        pe_time = repeat(pe_time, '1 ch -> ch h w', h=self.encoder_dim, w=self.encoder_dim)
        pe = torch.cat((pe_spatial, pe_time), dim=0)
        pe = rearrange(pe, 'ch h w -> (h w) 1 ch')
        return pe[:, :, :self.d_model]

    def pe_strokes_tokens(self, pos, device):
        n_strokes, bs, _ = pos.shape
        pe_spatial, pe_time = self.positionalencoding3d(self.encoder_dim, self.encoder_dim, n_strokes, self.d_model,
                                                        device)
        pe_spatial = self.bilinear_sampling_length_first(pe_spatial, pos[:, :, :2])
        pe_time = repeat(pe_time, 'L ch -> L bs ch', bs=bs)
        pe = torch.cat((pe_spatial, pe_time), dim=-1)
        return pe[:, :, :self.d_model]

    def bilinear_sampling_length_first(self, feat, pos) :
        n_strokes, bs, _ = pos.shape
        feat_temp = repeat(feat, 'ch h w -> n_reps ch h w', n_reps=n_strokes * bs)
        grid = rearrange(pos, 'L bs p -> (L bs) 1 1 p')

        pooled_features = F.grid_sample(feat_temp, 2 * grid - 1, align_corners=False, mode='bicubic')
        pooled_features = rearrange(pooled_features, '(L bs) ch 1 1 -> L bs ch', L=n_strokes)

        return pooled_features


    def positionalencoding3d(self, x, y, z, orig_channels, device) :
        channels = int(np.ceil(orig_channels / 6) * 2)
        if channels % 2 :
            channels += 1
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))

        pos_x = torch.arange(x).type(inv_freq.type())
        pos_y = torch.arange(y).type(inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq)

        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)

        spatial_emb = torch.zeros((x, y, channels * 2))
        spatial_emb[:, :, :channels] = emb_x
        spatial_emb[:, :, channels :2 * channels] = emb_y
        spatial_emb = rearrange(spatial_emb, 'h w ch -> ch h w')

        # Time
        if z == 0 :
            time_emb = torch.zeros((channels, x, y))
        else :
            pos_z = torch.arange(z).type(inv_freq.type())
            sin_inp_z = torch.einsum("i,j->ij", pos_z, inv_freq)
            time_emb = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        return spatial_emb.to(device), time_emb.to(device)

########################################################################################################################

def positionalencoding1d(x, orig_channels):

    channels = orig_channels
    inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))

    pos_x = torch.arange(x).type(inv_freq.type())
    sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
    emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
    emb = torch.zeros((x, channels))
    emb[:, :channels] = emb_x

    return emb[None, :, :orig_channels]


########################################################################################################################
class PositionalEncoding:

    def __init__(self, config) :

        self.input_dim = config["dataset"]["resize"]
        self.encoder_dim = config["model"]["img_encoder"]["visual_feat_hw"]
        self.d_model = config["model"]["d_model"]

        # Parameters for PE
        self.channels = int(np.ceil(self.d_model / 6) * 2)
        if self.channels % 2 :
            self.channels += 1
        self.inv_freq = 1. / (10000 ** (torch.arange(0, self.channels, 2).float() / self.channels))

        # Store spatial PE
        self.spatial_pe = self._get_2d_pe(self.input_dim, self.input_dim)

    def _get_2d_pe(self, x, y) :
        pos_x = torch.arange(x)
        pos_y = torch.arange(y)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2))
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels :] = emb_y

        emb = rearrange(emb, 'h w ch -> 1 ch h w')
        return emb

    def _get_1d_pe(self, x) :
        pos_x = torch.arange(x)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        return emb

    def pe_visual_tokens(self, device) :
        k = int(self.input_dim / self.encoder_dim)
        spatial_pe = nn.AvgPool2d((k, k))(self.spatial_pe)
        time_pe = torch.zeros((1, (self.d_model - 2 * self.channels), self.encoder_dim, self.encoder_dim))
        pe = torch.cat((spatial_pe, time_pe), dim=1)
        pe = rearrange(pe, 'bs ch h w -> (h w) bs ch')
        return pe.to(device)

    def pe_strokes_tokens(self, pos, device) :
        pos = pos[:, :, :2]
        L, bs, _ = pos.shape

        # Spatial
        pe = torch.empty([L, bs, 2 * self.channels])
        for l in range(L) :
            for b in range(bs) :
                tmp = 2 * pos[l, b] - 1
                pe[l, b] = F.grid_sample(self.spatial_pe, tmp.reshape(1, 1, 1, 2).float().cpu(), align_corners=False,
                                         mode='nearest').squeeze()

        # Time
        time_pe = self._get_1d_pe(L)
        time_pe = repeat(time_pe, 'L ch -> L bs ch', bs=bs)

        # Cat
        pe = torch.cat((pe, time_pe), dim=-1)
        return pe[:, :, :self.d_model].to(device)