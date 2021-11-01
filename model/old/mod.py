import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange
import torch
import torch.nn as nn
from einops import rearrange, repeat
from model.networks.image_encoders import resnet18
from timm.models.layers import trunc_normal_


class PositionalEncoding(nn.Module) :
    def __init__(self, d_model, dropout=0.1, max_len=5000) :
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0 : :2] = torch.sin(position * div_term)
        pe[:, 1 : :2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x) :
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


#######################################################
class PE :

    def __init__(self, type, d_model, length_first=False, **kwargs) :
        assert type == '2d' or type == '1d'
        self.type = type
        self.d_model = d_model
        self.length_first = length_first

        if type == '2d':
            self.h = kwargs['h']
            self.w = kwargs['w']
        else:
            self.length = kwargs['length']

    def positionalencoding1d(self, d_model, length) :
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0 :
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0 : :2] = torch.sin(position.float() * div_term)
        pe[:, 1 : :2] = torch.cos(position.float() * div_term)

        return pe[None]

    def positionalencoding2d(self, d_model, height, width) :
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0 :
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0 :d_model :2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1 :d_model :2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model : :2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1 : :2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe[None]

    def __call__(self) :

        if self.type == '1d':
            pe = self.positionalencoding1d(d_model=self.d_model, length=self.length)
        else :
            pe = self.positionalencoding2d(d_model=self.d_model, height=self.h, width=self.w)
            pe = rearrange(pe, 'bs d_model h w -> bs (h w) d_model')

        return pe.detach()


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

        if config["model"]["encoder_pe"] == "sine" :
            self.pe2D = PE(type='2d', d_model=self.d_model, h=self.visual_features_size, w=self.visual_features_size)()
            self.pe1D_ctx = PE(type='1d', d_model=self.d_model, length=self.context_length)()
            self.pe1D_seq = PE(type='1d', d_model=self.d_model, length=self.seq_length)()
        else :
            # Learnable PE, length first
            self.pe2D = nn.Parameter(torch.zeros(1, self.visual_features_size ** 2, self.d_model))
            self.pe1D_ctx = nn.Parameter(torch.zeros(1, self.context_length, self.d_model))
            self.pe1D_seq = nn.Parameter(torch.zeros(1, self.seq_length, self.d_model))
            trunc_normal_(self.pe2D, std=0.02)
            trunc_normal_(self.pe1D_ctx, std=0.02)
            trunc_normal_(self.pe1D_seq, std=0.02)

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
        img_feat,_ = self.img_encoder(img)
        canvas_feat,_ = self.canvas_encoder(canvas)
        visual_feat = self.conv_proj(torch.cat((img_feat, canvas_feat), dim=1))

        visual_feat = visual_feat.reshape(bs, self.d_model, -1).permute(0, 2, 1)
        visual_feat += self.pe2D.to(device=visual_feat.device)
        #visual_feat += self.visual_token

        # Context
        strokes_ctx_feat = self.proj_features(strokes_ctx)
        strokes_ctx_feat += self.pe1D_ctx.to(device=strokes_ctx_feat.device)
        #strokes_ctx_feat += self.stroke_token

        ctx_sequence = torch.cat((visual_feat, strokes_ctx_feat), dim=1)

        # Sequence
        x_sequence = self.proj_features(strokes_seq)
        x_sequence += self.pe1D_seq.to(device=x_sequence.device)
        #x_sequence += self.stroke_token

        # Permute sequences as length-first
        ctx_sequence = ctx_sequence.permute(1, 0, 2)
        x_sequence = x_sequence.permute(1, 0, 2)

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

        if config["model"]["decoder_pe"] == 'sine':
            self.time_queries_PE = PE(type='1d', d_model=self.d_model, length=self.seq_length)()
            self.time_queries_PE = self.time_queries_PE.permute(1, 0, 2)
        else :
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

    def decode(self, size, z, context) :
        time_queries = repeat(self.time_queries_PE.to(device=z.device), 'L 1 d_model -> L bs d_model', bs=size[1])

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
        out = self.decode(seq.size(), z, context)  # z is the input, context comes from the other branch

        return out, mu, log_sigma

    @torch.no_grad()
    def sample(self, ctx, L=None) :
        if L is None :
            L = self.seq_length
        bs = ctx.size(1)
        # Sample z
        z = torch.randn(bs, self.d_model).to(self.device)
        preds = self.decode(size=(L, bs, self.d_model), z=z, context=ctx)
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
    def generate(self, data, no_context=False, no_z=False) :
        context, x = self.embedder(data)
        context_features = self.context_encoder(context)
        if no_context :  # zero out the context to check if the model benefit from it
            context_features = torch.randn_like(context_features, device=context_features.device)
        if no_z :
            predictions = self.transformer_vae.sample(ctx=context_features)
        else :
            predictions = self.transformer_vae(x, context_features)[0]
        return predictions