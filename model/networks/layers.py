import torch
import numpy as np
from einops import rearrange, repeat
import torch.nn.functional as F


########################################################################################################################
class PEWrapper :
    def __init__(self, config) :
        self.input_dim = config["dataset"]["resize"]
        self.encoder_dim = int(config["model"]["img_encoder"]["visual_feat_hw"] * self.input_dim / 256)
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
        grid = rearrange(pos, 'L bs p -> (L bs) 1 1 p').detach()

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