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


#######################################################
class PE :

    def __init__(self, type, d_model, length_first=False, **kwargs) :
        assert type == '2d' or type == '1d'
        self.type = type
        self.d_model = d_model
        self.length_first = length_first

        if type == '2d':
            self.h = kwargs['h']
            self.w = kwargs['w']
        else:
            self.length = kwargs['length']

    def positionalencoding1d(self, d_model, length) :
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0 :
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0 : :2] = torch.sin(position.float() * div_term)
        pe[:, 1 : :2] = torch.cos(position.float() * div_term)

        return pe[None]

    def positionalencoding2d(self, d_model, height, width) :
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0 :
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0 :d_model :2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1 :d_model :2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model : :2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1 : :2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe[None]

    def __call__(self) :

        if self.type == '1d':
            pe = self.positionalencoding1d(d_model=self.d_model, length=self.length)
        else :
            pe = self.positionalencoding2d(d_model=self.d_model, height=self.h, width=self.w)
            pe = rearrange(pe, 'bs d_model h w -> bs (h w) d_model')


        if self.length_first:
            pe = rearrange(pe, 'bs L d_model -> L bs d_model')

        return pe.detach()

