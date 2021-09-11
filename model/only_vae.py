import torch
import torch.nn as nn
from einops import rearrange, repeat
from model.networks.layers import PositionalEncoding


class TransformerVAE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = config["device"]
        self.s_params = config["model"]["n_strokes_params"]
        self.d_model = config["model"]["d_model"]
        self.seq_length = config["dataset"]["sequence_length"]
        self.context_length = config["dataset"]["context_length"]+1

        self.time_queries_PE = PositionalEncoding(self.d_model, max_len=100)

        #self.query_dec = nn.Parameter(torch.randn(self.seq_length, self.d_model))
        self.mu = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.log_sigma = nn.Parameter(torch.randn(1, 1, self.d_model))

        #
        self.proj_z_context = nn.Linear(2*self.d_model, self.d_model)

        # Define Encoder and Decoder
        self.vae_encoder = nn.TransformerEncoder(
                            encoder_layer=nn.TransformerEncoderLayer(
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
        if config["model"]["activation_last_layer"] == "sigmoid":
            self.prediction_head.add_module('act', nn.Sigmoid())


    def encode(self, x):
        bs = x.size(1)

        # add learnable tokens
        mu = repeat(self.mu, '1 1 dim -> 1 bs dim', bs=bs)
        log_sigma = repeat(self.log_sigma, '1 1 dim -> 1 bs dim', bs=bs)
        x = torch.cat((mu, log_sigma, x), dim=0)  # (T+2) x bs x d_model

        # Encode the input
        x = self.vae_encoder(x)
        mu = x[0]     # first element of the seq
        log_var = x[1]   # second element of the seq

        return mu, log_var

    def sample_latent_z(self, mu, log_sigma):

        sigma = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(sigma)

        z = eps.mul(sigma).add_(mu)
        return z

    def decode(self, size, z):
        time_queries = torch.zeros(size, device=self.device)
        time_queries = self.time_queries_PE(time_queries)

        out = self.vae_decoder(time_queries, z.unsqueeze(0))

        # Linear proj
        out = out.permute(1, 0, 2)  # bs x L x dim
        out = self.prediction_head(out)

        return out

    def forward(self, xseq):

        mu, log_sigma = self.encode(xseq)
        z = self.sample_latent_z(mu, log_sigma)
        out = self.decode(size=xseq.size(), z=z)

        return out, mu, log_sigma

    @torch.no_grad()
    def generate(self, L):
        z = torch.randn(1, self.d_model, device=self.device)
        preds = self.decode(size=(L, 1, self.d_model), z=z)
        return preds

# ----------------------------------------------------------------------------------------------------------------------

class OnlyVAE(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.feature_proj = nn.Linear(config["model"]["n_strokes_params"], config["model"]["d_model"])
        self.transformer_vae = TransformerVAE(config)

    def forward(self, data):

        xseq = data['strokes_seq']
        xseq = self.feature_proj(xseq)
        xseq = xseq.permute(1, 0, 2)  # length first
        predictions, mu, log_sigma = self.transformer_vae(xseq)

        return predictions, mu, log_sigma

    @torch.no_grad()
    def generate(self, data, L):
        # TODO: fix this, with onlyvae we don't need the data
        preds = self.transformer_vae.generate(L)
        return preds