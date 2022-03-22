import torch
import numpy as np
from einops import rearrange, repeat

########################################################################################################################
def positionalencoding1d(x, orig_channels, offset=None):

    channels = orig_channels
    inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))

    pos_x = torch.arange(x).type(inv_freq.type())
    if offset:
        pos_x += offset
    sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
    emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
    emb = torch.zeros((x, channels))
    emb[:, :channels] = emb_x

    return emb[None, :, :orig_channels]


########################################################################################################################
class PositionalEncoding:

    def __init__(self, config) :
        self.img_size = config["dataset"]["resize"]
        self.visual_features = config["model"]["img_encoder"]["visual_feat_hw"]
        self.d_model = config["model"]["d_model"]

        self.channels = int(np.ceil(self.d_model / 6) * 2)
        if self.channels % 2 :
            self.channels += 1
        self.inv_freq = 1. / (10000 ** (torch.arange(0, self.channels, 2).float() / self.channels))

    def pe_visual_tokens(self, device):
        start = (self.visual_features / 2)
        end = self.img_size - (self.visual_features / 2)
        grid = torch.linspace(start=start, end=end, steps=self.visual_features)
        pos_y, pos_x = torch.meshgrid(grid, grid)
        pos_y = rearrange(pos_y, 'h w -> (h w)')
        pos_x = rearrange(pos_x, 'h w -> (h w)')
        pos_z = None

        output = self.positionalencoding3d(pos_x, pos_y, pos_z).to(device)
        return output.unsqueeze(dim=1)  # Unsqueeze batch dimension

    def pe_strokes_tokens(self, pos, device, offset_z = None):
        pos = pos.cpu().detach()
        n_strokes, bs, dim = pos.shape
        if dim == 8:
            pos = pos[:, :, :2]
        pos = rearrange(pos, 'n_strokes bs dim -> (n_strokes bs) dim')
        pos_x, pos_y = torch.split(pos, 1, dim=-1)
        pos_x = (pos_x * self.img_size).squeeze()
        pos_y = (pos_y * self.img_size).squeeze()
        pos_z = repeat(torch.arange(n_strokes), 'n_strokes -> (n_strokes bs)', bs=bs)
        if offset_z:
            pos_z += offset_z

        output = self.positionalencoding3d(pos_x, pos_y, pos_z)
        output = rearrange(output, '(n_strokes bs) dim -> n_strokes bs dim', n_strokes=n_strokes)
        return output.to(device)

    def positionalencoding3d(self, pos_x, pos_y, pos_z=None) :

        n = pos_x.shape[0]

        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)

        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)

        spatial_emb = torch.zeros((n, self.channels * 2))
        spatial_emb[:, :self.channels] = emb_x
        spatial_emb[:, self.channels: ] = emb_y

        # Time
        if pos_z is None:
            time_emb = torch.zeros((n, self.channels))
        else :
            sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
            time_emb = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)

        pe_3d = torch.cat((spatial_emb, time_emb), dim=-1)
        return pe_3d[:, :self.d_model]