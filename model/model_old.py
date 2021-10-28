import torch
import torch.nn as nn
from einops import rearrange, repeat
from model.networks.image_encoders import resnet18
from timm.models.layers import trunc_normal_
import numpy as np

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
        if self.pe_type == '3dsine':
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


def count_parameters(net) :
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


class Embedder(nn.Module) :

    def __init__(self, config) :
        super(Embedder, self).__init__()

        self.s_params = config["model"]["n_strokes_params"]
        self.d_model = config["model"]["d_model"]
        self.context_length = config["dataset"]["context_length"]
        self.seq_length = config["dataset"]["sequence_length"]
        self.visual_features_size = config["model"]["img_encoder"]["visual_features_dim"]

        self.PE = PEWrapper(config)
        self.visual_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.stroke_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        trunc_normal_(self.visual_token, std=0.02)
        trunc_normal_(self.stroke_token, std=0.02)

        self.img_encoder = resnet18(pretrained=config["model"]["img_encoder"]["pretrained"],
                                    layers_to_remove=config["model"]["img_encoder"]["layers_to_remove"])
        self.canvas_encoder = resnet18(pretrained=config["model"]["img_encoder"]["pretrained"],
                                       layers_to_remove=config["model"]["img_encoder"]["layers_to_remove"])

        self.conv_proj = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding=1, stride=1)
        self.proj_features = nn.Linear(self.s_params, self.d_model)

    def forward(self, data) :
        strokes_seq = data['strokes_seq']
        strokes_ctx = data['strokes_ctx']
        img = data['img']
        canvas = data['canvas']
        bs = img.shape[0]

        # Encode Img/Canvas
        img_feat, _ = self.img_encoder(img)
        canvas_feat, _ = self.canvas_encoder(canvas)
        visual_feat = self.conv_proj(torch.cat((img_feat, canvas_feat), dim=1))
        visual_feat = rearrange(visual_feat, 'bs dim h w -> bs h w dim')  # channels last

        # Strokes
        strokes = torch.cat((strokes_ctx, strokes_seq), dim=1)
        strokes_feat = self.proj_features(strokes)

        # Add 3D PE
        visual_feat, strokes_feat = self.PE(visual_feat, strokes_feat, strokes)
        visual_feat = rearrange(visual_feat, 'bs h w dim -> bs (h w) dim')

        visual_feat += self.visual_token
        strokes_feat += self.stroke_token

        # Rearrange
        strokes_ctx_feat = strokes_feat[:, :self.context_length, :]
        strokes_seq_feat = strokes_feat[:, self.context_length :, :]

        # Context
        ctx_sequence = torch.cat((visual_feat, strokes_ctx_feat), dim=1)

        # Length first
        ctx_sequence = rearrange(ctx_sequence, 'bs L dim -> L bs dim')
        x_sequence = rearrange(strokes_seq_feat, 'bs L dim -> L bs dim')

        return ctx_sequence, x_sequence


# ----------------------------------------------------------------------------------------------------------------------

class ContextEncoder(nn.Module) :

    def __init__(self, config) :
        super().__init__()

        self.net = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=config["model"]["d_model"],
                nhead=config["model"]["encoder"]["n_heads"],
                dim_feedforward=config["model"]["encoder"]["ff_dim"],
                activation=config["model"]["encoder"]["act"],
            ),
            num_layers=config["model"]["encoder"]["n_layers"])

    def forward(self, x) :
        x = self.net(x)
        return x


# ----------------------------------------------------------------------------------------------------------------------

class TransformerVAE(nn.Module) :
    def __init__(self, config) :
        super().__init__()

        self.device = config["device"]
        self.s_params = config["model"]["n_strokes_params"]
        self.d_model = config["model"]["d_model"]
        self.seq_length = config["dataset"]["sequence_length"]
        self.context_length = config["dataset"]["context_length"]

        self.ctx_z = config["model"]["ctx_z"]  # how to merge context and z
        if self.ctx_z == 'proj' :
            self.proj_ctx_z = nn.Linear(2 * self.d_model, self.d_model)

        self.time_queries_PE = nn.Parameter(torch.zeros(self.seq_length, 1, self.d_model))
        trunc_normal_(self.time_queries_PE, std=0.02)

        self.mu = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.log_sigma = nn.Parameter(torch.randn(1, 1, self.d_model))

        # Define Encoder and Decoder
        self.vae_encoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=config["model"]["vae_encoder"]["n_heads"],
                dim_feedforward=config["model"]["vae_encoder"]["ff_dim"],
                activation=config["model"]["vae_encoder"]["act"],
            ),
            num_layers=config["model"]["vae_encoder"]["n_layers"])

        self.vae_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=config["model"]["vae_decoder"]["n_heads"],
                dim_feedforward=config["model"]["vae_decoder"]["ff_dim"],
                activation=config["model"]["vae_decoder"]["act"],
            ),
            num_layers=config["model"]["vae_decoder"]["n_layers"])

        # Final projection head
        self.prediction_head = nn.Sequential(
            nn.Linear(self.d_model, self.s_params))
        if config["model"]["activation_last_layer"] == "sigmoid" :
            self.prediction_head.add_module('act', nn.Sigmoid())

    def encode(self, x, context) :
        bs = x.size(1)

        # add learnable tokens
        mu = repeat(self.mu, '1 1 dim -> 1 bs dim', bs=bs)
        log_sigma = repeat(self.log_sigma, '1 1 dim -> 1 bs dim', bs=bs)
        x = torch.cat((mu, log_sigma, x), dim=0)  # (T+2) x bs x d_model

        # Encode the input
        x = self.vae_encoder(x, context)
        mu = x[0]  # first element of the seq
        log_var = x[1]  # second element of the seq

        return mu, log_var

    def reparameterize(self, mu, log_sigma) :

        sigma = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(sigma)

        z = eps.mul(sigma).add_(mu)
        return z

    def decode(self, z, context) :
        bs = z.size(0)
        time_queries = repeat(self.time_queries_PE.to(device=z.device), 'L 1 d_model -> L bs d_model', bs=bs)

        if self.ctx_z == 'proj' :
            # # Fuse z and context using projection
            z = repeat(z, 'bs n_feat -> ctx_len bs n_feat', ctx_len=self.context_length)
            z_ctx = torch.cat([context, z], dim=-1)  # concatenate on the feature dimension
            z_ctx = self.proj_ctx_z(z_ctx)
        elif self.ctx_z == 'cat' :
            # Fuse z and context with concatenation on length dim
            z_ctx = torch.cat((context, z[None]), dim=0)
        else :
            raise NotImplementedError()

        out = self.vae_decoder(time_queries, z_ctx)

        # Linear proj
        out = out.permute(1, 0, 2)  # bs x L x dim
        out = self.prediction_head(out)

        return out

    def forward(self, seq, context) :

        mu, log_sigma = self.encode(seq, context)
        z = self.reparameterize(mu, log_sigma)

        # Replicate z and decode
        out = self.decode(z, context)  # z is the input, context comes from the other branch

        return out, mu, log_sigma

    @torch.no_grad()
    def sample(self, ctx, L=None) :
        if L is None :
            L = self.seq_length
        bs = ctx.size(1)
        # Sample z
        z = torch.randn(bs, self.d_model).to(self.device)
        preds = self.decode(z=z, context=ctx)
        return preds


# ----------------------------------------------------------------------------------------------------------------------

class InteractivePainter(nn.Module) :

    def __init__(self, config) :
        super().__init__()

        self.embedder = Embedder(config)
        self.context_encoder = ContextEncoder(config)
        self.transformer_vae = TransformerVAE(config)

    def forward(self, data) :

        context, x = self.embedder(data)
        context_features = self.context_encoder(context)
        predictions, mu, log_sigma = self.transformer_vae(x, context_features)

        return predictions, mu, log_sigma

    @torch.no_grad()
    def generate(self, data, no_context=False, no_z=True) :
        context, x = self.embedder(data)
        context_features = self.context_encoder(context)
        if no_context :  # zero out the context to check if the model benefit from it
            context_features = torch.randn_like(context_features, device=context_features.device)
        if no_z :
            predictions = self.transformer_vae.sample(ctx=context_features)
        else :
            predictions = self.transformer_vae(x, context_features)[0]
        return predictions


if __name__ == '__main__' :
    from dataset import StrokesDataset
    from torch.utils.data import DataLoader
    from utils.parse_config import ConfigParser
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default='a')
    parser.add_argument("--config",
                        default='/Users/eliap/Projects/brushstrokes-generation/configs/train/config_local.yaml')
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()

    c_parser = ConfigParser(args)
    c_parser.parse_config(args)
    config = c_parser.get_config()

    # dataset = StrokesDataset(config=config, isTrain=True)

    # dataloader = DataLoader(dataset, batch_size=2)
    # data = next(iter(dataloader))

    data = {
        'strokes_ctx' : torch.rand((3, 10, 11)),
        'strokes_seq' : torch.rand((3, 8, 11)),
        'canvas' : torch.randn((3, 3, 256, 256)),
        'img' : torch.randn((3, 3, 256, 256))
    }

    # Define the model
    net = InteractivePainter(config)


    # preds = net(data)

    def count_parameters(model) :
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    params = count_parameters(net)
    print(f'Number of trainable parameters: {params / 10 ** 6}')

    # Predict with context
    net.train()
    clean_preds = net(data)
    v = net.generate(data, no_z=True)

    print(v.shape)