import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_
from .layers import PEWrapper, PositionalEncoding


def reparameterize(mu, log_sigma):
    sigma = log_sigma.mul(0.5).exp_()
    eps = torch.randn_like(sigma)
    z = eps.mul(sigma).add_(mu)

    return z

class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()

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
        self.visual_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.stroke_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        trunc_normal_(self.visual_token, std=0.02)
        trunc_normal_(self.stroke_token, std=0.02)

        # Learnable tokens mu / sigma
        self.mu = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.log_sigma = nn.Parameter(torch.randn(1, 1, self.d_model))
        trunc_normal_(self.mu, std=0.02)
        trunc_normal_(self.log_sigma, std=0.02)

        # Define Encoder and Decoder
        self.encoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=config["model"]["encoder"]["n_heads"],
                dim_feedforward=config["model"]["encoder"]["ff_dim"],
                activation=config["model"]["encoder"]["act"],
            ),
            num_layers=config["model"]["encoder"]["n_layers"])

    def forward(self, data, context):

        # Project strokes
        strokes_seq = data['strokes_seq']
        strokes_seq = rearrange(strokes_seq, 'bs L dim -> L bs dim')

        x_sequence = self.proj_features(strokes_seq)
        x_sequence += self.PE.pe_strokes_tokens(pos=strokes_seq, device=x_sequence.device)
        x_sequence += self.stroke_token


        # Encoder
        bs = x_sequence.size(1)

        # add learnable tokens
        mu = repeat(self.mu, '1 1 dim -> 1 bs dim', bs=bs)
        log_sigma = repeat(self.log_sigma, '1 1 dim -> 1 bs dim', bs=bs)
        x = torch.cat((mu, log_sigma, x_sequence), dim=0)  # (T+2) x bs x d_model

        # Encode the input
        x = self.encoder(x, context)
        mu = x[0]  # first element of the seq
        log_sigma = x[1]  # second element of the seq
        z = reparameterize(mu, log_sigma)
        return z, mu, log_sigma