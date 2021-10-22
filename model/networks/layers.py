import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange


class PositionalEncoding(nn.Module) :
    def __init__(self, d_model, dropout=0.1, max_len=5000) :
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0 : :2] = torch.sin(position * div_term)
        pe[:, 1 : :2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x) :
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

########################################################################################################################
class PEWrapper:
    def __init__(self, config):
        self.pe_type = config["model"]["encoder_pe"]

    def pe3d(self, visual_features, strokes_features, strokes_params):
        assert visual_features.size(-1) == strokes_features.size(-1)

        h = visual_features.size(2)
        bs, length, dim = strokes_features.shape

        pe3d = positionalencoding3d(size=(h, h, length+1, dim))

        # Visual Features
        visual_features += pe3d[:, :, :, 0, :].to(device=visual_features.device)

        # Strokes Features
        x_i, y_i = pos_2_idx(strokes_params[:, :, :2])   # (x,y)
        pe_strokes = torch.empty_like(strokes_features)
        for b_idx in range(bs):
            for l_idx in range(length):
                pe_strokes[b_idx, l_idx] = pe3d[0, y_i[b_idx, l_idx], x_i[b_idx, l_idx], l_idx+1, :]
        strokes_features += pe_strokes.to(device=strokes_features.device)

        return visual_features, strokes_features

    def pe_old(self, visual_features, strokes_features):
        assert visual_features.size(-1) == strokes_features.size(-1)

        h = visual_features.size(2)
        bs, length, dim = strokes_features.shape

        pe2d = positionalencoding2d(size=(h, h, dim))
        pe1d = positionalencoding1d(size=(length, dim))

        # Visual Features
        visual_features += pe2d.to(device=visual_features.device)

        # Strokes Features
        strokes_features += pe1d.to(device=strokes_features.device)

        return visual_features, strokes_features


    def __call__(self, visual_features, strokes_features, strokes_params):
        if self.pe_type == '3d_sine':
            return self.pe3d(visual_features, strokes_features, strokes_params)
        elif self.pe_type == 'sine':
            return self.pe_old(visual_features, strokes_features)
        else:
            raise NotImplementedError()

def pos_2_idx(pos):
    width = 256
    vis = 8
    n = int(256 / 8)
    with torch.no_grad():
        pos = pos.detach().cpu().numpy()
        pos = np.rint(pos * (width - 1) + 0.5)
    x_i = (pos[:, :, 0] // n).astype('int')
    y_i = (pos[:, :, 1] // n).astype('int')
    return x_i, y_i


def positionalencoding1d(size):
    x, orig_channels = size

    channels = orig_channels
    inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))

    pos_x = torch.arange(x).type(inv_freq.type())
    sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
    emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
    emb = torch.zeros((x, channels))
    emb[:, :channels] = emb_x

    return emb[None, :, :orig_channels].detach()


def positionalencoding2d(size):
    x, y, orig_channels = size

    channels = int(np.ceil(orig_channels / 4) * 2)
    channels = channels
    inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))

    pos_x = torch.arange(x).type(inv_freq.type())
    pos_y = torch.arange(y).type(inv_freq.type())
    sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
    sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq)
    emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
    emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
    emb = torch.zeros((x, y, channels * 2))
    emb[:, :, :channels] = emb_x
    emb[:, :, channels :2 * channels] = emb_y

    return emb[None, :, :, :orig_channels].detach()

def positionalencoding3d(size):

    # Unpack size
    x, y, z, orig_channels = size

    channels = int(np.ceil(orig_channels / 6) * 2)
    if channels % 2 :
        channels += 1
    inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))

    pos_x = torch.arange(x).type(inv_freq.type())
    pos_y = torch.arange(y).type(inv_freq.type())
    pos_z = torch.arange(z).type(inv_freq.type())
    sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
    sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq)
    sin_inp_z = torch.einsum("i,j->ij", pos_z, inv_freq)
    emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
    emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
    emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
    emb = torch.zeros((x, y, z, channels * 3))
    emb[:, :, :, :channels] = emb_x
    emb[:, :, :, channels :2 * channels] = emb_y
    emb[:, :, :, 2 * channels :] = emb_z

    # Set firs channel of z axis to 0, used for visual tokens
    emb[:, :, 0, :] = torch.zeros((x, y, emb.size(3)))

    return emb[None, :, :, :, :orig_channels].detach()