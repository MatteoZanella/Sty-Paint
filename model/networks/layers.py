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

    def forward(self, x, idx=None):
        """
        :param x: tensor of size bs x length x features to add positional emebddings
        """
        if idx is None:
            return x + self.pos_table.clone().detach()
        else:
            return x + self.pos_table[:, idx:, :].clone().detach()

# ----------------------------------------------------------------------------------------------------------------------

class Merger(nn.Module):

    def __init__(self, config):
        super(Merger, self).__init__()

        self.type = config["model"]["merge_type"]
        self.n_strokes_params = config["model"]["n_strokes_params"]
        self.padding_value = 0

        if self.type == "sum":
            self.s_params_proj = nn.Linear(self.n_strokes_params, 512)


    def merge_strokes_canvas(self, strokes_params, canvas_feat):
        if self.type == "concat":
            out = torch.cat([canvas_feat, strokes_params], dim=2)  # concatenate on the feature dimension
            return out
        elif self.type == "sum":
            strokes_proj = self.s_params_proj(strokes_params)
            out = canvas_feat + strokes_proj
            return out
        else:
            raise Exception

    def pad_img_features(self, x):
        if self.type == 'concat':
            bs = x.size(0)
            x = torch.cat([x, torch.full([bs, self.n_strokes_params], self.padding_value,device=x.device)], dim=-1)
            return x.unsqueeze(1)
        elif self.type == 'sum':
            return x.unsqueeze(1)
        else:
            raise Exception


