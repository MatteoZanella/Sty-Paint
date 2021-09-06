import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x, offset=0):
        """
        :param x: tensor of size bs x length x features to add positional emebddings
        """
        pe = self.pos_table.clone().detach().to(x.device)
        return x[:, offset:, :] + pe

# ----------------------------------------------------------------------------------------------------------------------

class SequenceMerger(nn.Module):

    def __init__(self, config):
        super(SequenceMerger, self).__init__()

        self.d_model = config["model"]["d_model"]
        self.s_params = config["model"]["n_strokes_params"]

        self.img_proj = nn.Linear(512, self.d_model)  # project image features to lower dimension
        self.seq_proj = nn.Linear(512 + self.s_params, self.d_model)


    def forward(self, strokes_params, canvas_feat, img_feat=None):
        # Project img features
        if img_feat is not None:
            img_feat = self.img_proj(img_feat)
            img_feat = img_feat.unsqueeze(1)

        # Concatenate canvas and strokes and project
        x = torch.cat([canvas_feat, strokes_params], dim=-1)  # concatenate along feature dim
        x = self.seq_proj(x)

        if img_feat is not None:
            return torch.cat([img_feat, x], dim=1)  # concatenate along length dim
        else:
            return x

