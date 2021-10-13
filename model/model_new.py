import torch
import torch.nn as nn
from einops import rearrange, repeat
from networks.image_encoders import resnet18
from networks.layers import PE
# from model.networks.image_encoders import ResNetEncoder, ConvEncoder
# from model.networks.layers import PositionalEncoding
from timm.models.layers import trunc_normal_

def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

class Embedder(nn.Module) :

    def __init__(self, config) :
        super(Embedder, self).__init__()

        self.s_params = config["model"]["n_strokes_params"]
        self.d_model = config["model"]["d_model"]
        self.context_length = config["dataset"]["context_length"]
        self.seq_length = config["dataset"]["sequence_length"]

        if config["model"]["encoder"]["pe"] == "sine":
            self.pe2D = PE(type='2d', d_model=self.d_model, h=8, w=8)
            self.pe1D = PE(type='1d', d_model=self.d_model, length=self.context_length)
        else:
            # Learnable PE, length first
            self.pe2D = nn.Parameter(torch.zeros(1 , 8*8, self.d_model))
            self.pe1D = nn.Parameter(torch.zeros(1, self.context_length, self.d_model))
            trunc_normal_(self.pe2D, std=0.02)
            trunc_normal_(self.pe1D, std=0.02)

        self.img_encoder = resnet18()

        print(f'Conv encoder {count_parameters(self.img_encoder) / 10 ** 6}')
        self.proj_features = nn.Linear(self.s_params, self.d_model)

    def forward(self, data) :
        strokes_seq = data['strokes_seq']
        strokes_ctx = data['strokes_ctx']
        img = data['img']
        canvas = data['canvas']
        bs = img.shape[0]

        # Encode Img/Canvas
        img_feat = self.img_encoder(img)
        img_feat = img_feat.reshape(bs, 256, -1).permute(0, 2, 1)
        img_feat += self.pe2D

        canvas_feat = self.img_encoder(canvas)
        canvas_feat = canvas_feat.reshape(bs, 256, -1).permute(0, 2, 1)
        canvas_feat += self.pe2D

        # Context
        ctx_sequence = self.proj_features(strokes_ctx)
        ctx_sequence += self.pe1D

        # Sequence
        x_sequence = self.proj_features(strokes_seq)
        #x_sequence = self.pe1D(x=x_sequence,  dmodel=256, length=self.seq_length)

        # Add positional encodings to the sequences
        ctx_sequence = torch.cat((canvas_feat, canvas_feat, ctx_sequence), dim=1)

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

        self.pool_context = config["model"]["encoder"]["context_pooling"]
        if self.pool_context :
            self.context_token = nn.Parameter(torch.randn(1, 1, config["model"]["d_model"]))

    def forward(self, x) :
        if self.pool_context :
            fake_ctx = repeat(self.context_token, '1 1 dim -> 1 bs dim', bs=x.size(1))
            x = torch.cat((fake_ctx, x), dim=0)  # add fake token
        x = self.net(x)

        if self.pool_context :
            return x[0][None]  # return only the fake token, summarizing the whole sequence.
        else :
            return x  # retrun the whole sequence


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

        self.time_queries_PE = nn.Parameter(torch.randn(self.seq_length, 1, self.d_model))
        # self.query_dec = nn.Parameter(torch.randn(self.seq_length, self.d_model))
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
        #time_queries = torch.zeros(size=(size[1], size[0], size[2]), device=self.device)
        time_queries = repeat(self.time_queries_PE, 'L 1 d_model -> L bs d_model', bs=size[1])  # Note, we won't be able to generate longer sequences

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
        z = torch.randn(bs, self.d_model).cuda()
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

if __name__ == '__main__':
    from dataset import StrokesDataset
    from torch.utils.data import DataLoader
    from utils.parse_config import ConfigParser
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default='a')
    parser.add_argument("--config", default='/Users/eliap/Projects/brushstrokes-generation/configs/train/config_local.yaml')
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()

    c_parser = ConfigParser(args)
    c_parser.parse_config(args)
    config = c_parser.get_config()

    #dataset = StrokesDataset(config=config, isTrain=True)

    #dataloader = DataLoader(dataset, batch_size=2)
    #data = next(iter(dataloader))

    data = {
        'strokes_ctx' : torch.randn((1,4, 12)),
        'strokes_seq' : torch.randn((1,8, 12)),
        'canvas' : torch.randn((1,3,256, 256)),
        'img' : torch.randn((1,3, 256, 256))
    }

    # Define the model
    net = InteractivePainter(config)

    #preds = net(data)


    def count_parameters(model) :
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    params = count_parameters(net)
    print(f'Number of trainable parameters: {params / 10 ** 6}')

    # Predict with context
    net.train()
    clean_preds = net(data)

