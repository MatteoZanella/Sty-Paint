import torch
import torch.nn as nn
from einops import rearrange, repeat
from networks.image_encoders import resnet18
from networks.layers import PE
from timm.models.layers import trunc_normal_
import numpy as np

def count_parameters(net) :
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

def pos_to_idx(x, width=256):
    with torch.no_grad():
        x = x.detach().numpy()
        x = np.rint(x * (width - 1) + 0.5)

        # index to feature maps
        x_i = x[:, :, 0] // 32
        y_i = x[:, :, 1] // 32

        flat_idx = y_i * 8 + x_i

    return flat_idx.transpose([1,0])

class Embedder(nn.Module) :

    def __init__(self, config) :
        super(Embedder, self).__init__()

        self.s_params = config["model"]["n_strokes_params"]
        self.d_model = config["model"]["d_model"]
        self.context_length = config["dataset"]["context_length"]
        self.seq_length = config["dataset"]["sequence_length"]

        if config["model"]["encoder_pe"] == "sine" :
            self.pe2D = PE(type='2d', d_model=self.d_model, h=8, w=8)()
            self.pe1D_ctx = PE(type='1d', d_model=self.d_model, length=self.context_length)()
            self.pe1D_seq = PE(type='1d', d_model=self.d_model, length=self.seq_length)()
        else :
            # Learnable PE, length first
            self.pe2D = nn.Parameter(torch.zeros(1, 8 * 8, self.d_model))
            self.pe1D_ctx = nn.Parameter(torch.zeros(1, self.context_length, self.d_model))
            self.pe1D_seq = nn.Parameter(torch.zeros(1, self.seq_length, self.d_model))
            trunc_normal_(self.pe2D, std=0.02)
            trunc_normal_(self.pe1D_ctx, std=0.02)
            trunc_normal_(self.pe1D_seq, std=0.02)

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
        img_feat = self.img_encoder(img)
        canvas_feat = self.canvas_encoder(canvas)
        visual_feat = self.conv_proj(torch.cat((img_feat, canvas_feat), dim=1))

        visual_feat = visual_feat.reshape(bs, self.d_model, -1).permute(0, 2, 1)
        visual_feat += self.pe2D.to(device=visual_feat.device)

        # Context
        strokes_ctx_feat = self.proj_features(strokes_ctx)
        strokes_ctx_feat += self.pe1D_ctx.to(device=strokes_ctx_feat.device)

        ctx_sequence = torch.cat((visual_feat, strokes_ctx_feat), dim=1)

        # Sequence
        x_sequence = self.proj_features(strokes_seq)
        x_sequence += self.pe1D_seq.to(device=x_sequence.device)

        # Permute sequences as length-first
        ctx_sequence = ctx_sequence.permute(1, 0, 2)
        x_sequence = x_sequence.permute(1, 0, 2)

        return ctx_sequence, x_sequence, visual_feat


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

        if config["model"]["decoder_pe"] == 'sine' :
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

        # Divide the decoder in 2 modules
        self.pos_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=config["model"]["vae_decoder"]["n_heads"],
                dim_feedforward=config["model"]["vae_decoder"]["ff_dim"],
                activation=config["model"]["vae_decoder"]["act"],
            ),
            num_layers=config["model"]["vae_decoder"]["n_layers"])

        self.pos_head = nn.Sequential(
            nn.Linear(self.d_model, 2),
            nn.Sigmoid())


        self.color_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=config["model"]["vae_decoder"]["n_heads"],
                dim_feedforward=config["model"]["vae_decoder"]["ff_dim"],
                activation=config["model"]["vae_decoder"]["act"],
            ),
            num_layers=config["model"]["vae_decoder"]["n_layers"])

        self.color_head = nn.Sequential(
            nn.Linear(self.d_model, 9),
            nn.Sigmoid())


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

    def decode_position(self, size, context):
        time_queries = repeat(self.time_queries_PE.to(device=context.device), 'L 1 d_model -> L bs d_model', bs=size[1])
        hidden = self.pos_decoder(time_queries, context)
        pos = self.pos_head(hidden)

        return pos

    def decode_color(self, pos, visual_features):
        flat_idx = pos_to_idx(pos, width=256)
        pooled_features = visual_features[flat_idx, :]

        return x_i




    def decode(self, size, z, context, visual_features) :
        time_queries = repeat(self.time_queries_PE.to(device=z.device), 'L 1 d_model -> L bs d_model', bs=size[1])
        z_ctx = torch.cat((context, z[None]), dim=0)

        pred_pos = self.decode_position(size, z_ctx)
        x_i, y_i = self.decode_color(pred_pos, visual_features)


        out = self.vae_decoder(time_queries, z_ctx)

        # Linear proj
        out = out.permute(1, 0, 2)  # bs x L x dim
        out = self.prediction_head(out)

        return out

    def forward(self, seq, context, visual_features) :

        mu, log_sigma = self.encode(seq, context)
        z = self.reparameterize(mu, log_sigma)

        # Replicate z and decode
        out = self.decode(seq.size(), z, context, visual_features)  # z is the input, context comes from the other branch

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

        context, x, vs_feat = self.embedder(data)
        context_features = self.context_encoder(context)
        predictions, mu, log_sigma = self.transformer_vae(x, context_features, vs_feat)

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
        'strokes_ctx' : torch.randn((2, 4, 12)),
        'strokes_seq' : torch.randn((2, 8, 12)),
        'canvas' : torch.randn((2, 3, 256, 256)),
        'img' : torch.randn((2, 3, 256, 256))
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
