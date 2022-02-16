import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_
from .layers import PEWrapper, PositionalEncoding
import functools



class TransformerDiscriminator(nn.Module):
    def __init__(self, config):
        super(TransformerDiscriminator, self).__init__()

        self.device = config["device"]
        self.s_params = config["model"]["n_strokes_params"]
        self.d_model = config["model"]["d_model"]
        self.seq_length = config["dataset"]["sequence_length"]
        self.context_length = config["dataset"]["context_length"]

        self.proj_features = nn.Linear(self.s_params, self.d_model)

        if config["model"]["encoder_pe"] == "new":
            print('Using new encodings')
            self.PE = PositionalEncoding(config)
        else:
            self.PE = PEWrapper(config)


        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        trunc_normal_(self.cls_token, std=0.02)

        # Number of blocks
        self.num_layers = config["model"]["discriminator"]["num_layers"]

        # If set, the discriminator takes as input only the sequence of strokes to classify
        self.encoder_only = config["model"]["discriminator"]["encoder_only"]

        if config["model"]["discriminator"]["encoder_only"]:
            self.net = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=config["model"]["decoder"]["n_heads"],
                    dim_feedforward=config["model"]["decoder"]["ff_dim"],
                    activation=config["model"]["decoder"]["act"],
                    dropout=config["model"]["dropout"]
                ),
                num_layers=self.num_layers)
        else:
            self.net = nn.TransformerDecoder(
                decoder_layer=nn.TransformerDecoderLayer(
                    d_model=self.d_model,
                    nhead=config["model"]["decoder"]["n_heads"],
                    dim_feedforward=config["model"]["decoder"]["ff_dim"],
                    activation=config["model"]["decoder"]["act"],
                    dropout=config["model"]["dropout"]
                ),
                num_layers=self.num_layers)

        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, 1)


    def forward(self, x, context):
        # Project strokes
        x = rearrange(x, 'bs L dim -> L bs dim')

        x = self.proj_features(x)
        x += self.PE.pe_strokes_tokens(pos=x, device=x.device)

        # add learnable tokens
        cls_token = repeat(self.cls_token, '1 1 dim -> 1 bs dim', bs=x.size(1))
        x = torch.cat((cls_token, x), dim=0)  # (T+2) x bs x d_model

        # Encode the input
        if self.encoder_only:
            x = self.net(x)
        else:
            x = self.net(x, context)

        x = self.norm(x[0])  # grab class token
        x = self.head(x)

        return x

# =========================================================================================
class Conv1dDiscriminator(nn.Module):

    def __init__(self, config) :
        super(Conv1dDiscriminator, self).__init__()
        self.s_params = config["model"]["n_strokes_params"]
        self.seq_length = config["dataset"]["sequence_length"]
        self.contex_length = config["dataset"]["context_length"]
        self.num_layers = config["model"]["discriminator"]["num_layers"]


        self.net = nn.Sequential(

            nn.Conv1d(in_channels=8,
                      out_channels=32,
                      kernel_size=5,
                      padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=32,
                      out_channels=64,
                      kernel_size=5,
                      padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)

        )
        self.m = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(256, 1)


    def forward(self, x, context):
        x = torch.cat((context, x), dim=1)
        x = rearrange(x, 'bs L dim -> bs dim L')
        x = self.net(x)
        x = self.m(x).squeeze()
        x = self.head(x)
        return x
#######################################################################
# Patch GAN Discriminator, from: https://github.com/junyanz/BicycleGAN/blob/master/models/networks.py

class D_NLayers(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(D_NLayers, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
# =========================================================================================
class Discriminator(nn.Module):

    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.disc_type = config["model"]["discriminator"]["type"]
        if self.disc_type == 'transformer':
            self.net = TransformerDiscriminator(config)
        elif self.disc_type == 'conv1d':
            self.net = Conv1dDiscriminator(config)
        elif self.disc_type == 'patch_gan':
            self.net = D_NLayers(n_layers=config["model"]["discriminator"]["num_layers"], input_nc=9)
        else:
            raise NotImplementedError()

    def forward(self, x, context=None):
        if self.disc_type == "transformer":
            return self.net(x, context)
        elif self.disc_type == "conv1d":
            return self.net(x, context)
        elif self.disc_type == 'patch_gan':
            return self.net(x)
        else:
            print(self.disc_type)
            raise NotImplementedError()
