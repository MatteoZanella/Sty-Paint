import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_
from .layers import PEWrapper, PositionalEncoding

class Discriminator(nn.Module):

    def __init__(self, config):
        super(Discriminator, self).__init__()

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

        # Define Encoder and Decoder
        self.net = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=config["model"]["decoder"]["n_heads"],
                dim_feedforward=config["model"]["decoder"]["ff_dim"],
                activation=config["model"]["decoder"]["act"],
            ),
            num_layers=config["model"]["decoder"]["n_layers"] // 2)

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
        x = self.net(x, context)
        x = self.norm(x[0])  # grab class token
        x = self.head(x)

        return x