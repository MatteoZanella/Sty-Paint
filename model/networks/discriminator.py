import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_
from .layers import PEWrapper, PositionalEncoding



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
        self.num_layers = config["model"]["discriminator"]["num_layers"]

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=self.s_params * (2 ** i),
                          out_channels=self.s_params * (2 ** (i + 1)),
                          kernel_size=3,
                          padding=1),
                nn.BatchNorm1d(self.s_params * (2 ** (i + 1))))
            for i in range(self.num_layers)])

        self.head = nn.Conv1d(in_channels=self.s_params * (2 ** (self.num_layers)),
                              out_channels=1,
                              kernel_size=self.seq_length)


    def forward(self, x):
        x = rearrange(x, 'bs L dim -> bs dim L')
        for blk in self.blocks:
            x = blk(x)
        x = self.head(x)
        return x.squeeze(1)

# =========================================================================================
class Discriminator(nn.Module):

    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.disc_type = config["model"]["discriminator"]["type"]
        if self.disc_type == 'transformer':
            self.net = TransformerDiscriminator(config)
        elif self.disc_type == 'conv1d':
            self.net = Conv1dDiscriminator(config)
        else:
            raise NotImplementedError()

    def forward(self, x, context):
        if self.disc_type == "transformer":
            return self.net(x, context)
        elif self.disc_type == "conv1d":
            return self.net(x)
        else:
            raise NotImplementedError()
