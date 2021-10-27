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
        self.input_dim = 256   # dimension of the image
        self.channels = 256
        self.encoder_dim = 8


    def pe_visual_tokens(self, x):

        hw = x.shape[-1]
        k = int(self.input_dim / hw)
        pe = positionalencoding3d(x=self.input_dim, y=self.input_dim, z=1, orig_channels=self.channels)
        pe = pe.permute(0, 4, 1, 2, 3)[:, :, :, :, 0]   # bs x ch x h x w
        pe = nn.AvgPool2d((k, k))(pe)

        return x + pe.to(x.device)

    def pe_strokes_tokens(self, x, params):
        pos = params[:, :, :2]
        idxs = torch.round(pos * (self.input_dim - 1) + 0.5).long()    # [0,1] -> [0, input_dim]
        pe = positionalencoding3d(x=self.input_dim, y=self.input_dim, z=x.size(1), orig_channels=self.channels)
        pe = pe[0, idxs[:, :, 1], idxs[:, :, 0], torch.arange(x.size(1)), :]         # bs x L x ch

        return x + pe.to(x.device)

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

    return emb[None, :, :, :orig_channels]

def positionalencoding3d(x,y,z,orig_channels):

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

    return emb[None, :, :, :, :orig_channels]