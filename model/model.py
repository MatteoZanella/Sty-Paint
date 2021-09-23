import torch
import torch.nn as nn
from einops import rearrange, repeat
from model.networks.image_encoders import ResNetEncoder
from model.networks.layers import PositionalEncoding


class Embedder(nn.Module):

    def __init__(self, config):
        super(Embedder, self).__init__()

        self.s_params = config["model"]["n_strokes_params"]
        self.d_model = config["model"]["d_model"]
        self.context_length = config["dataset"]["context_length"]
        self.seq_length = config["dataset"]["sequence_length"]
        #self.L = config["dataset_acquisition"]["total_length"]

        # Networks
        self.img_encoder = ResNetEncoder(pretrained=config["model"]["img_encoder"]["pretrained"])

        self.context_PE = PositionalEncoding(self.d_model, dropout=0)
        self.sequence_PE = PositionalEncoding(self.d_model, dropout=0)

        #self.context_PE = nn.Parameter(torch.randn(self.context_length+1, self.d_model))
        #self.sequence_PE = nn.Parameter(torch.randn(self.seq_length, self.d_model))

        self.proj_features = nn.Linear(512 + self.s_params, self.d_model)

    def encode_canvas(self, x):
        L = x.size(1)
        x = rearrange(x, 'bs L c h w -> (bs L) c h w')
        x = self.img_encoder(x)
        x = rearrange(x, '(bs L) n_feat -> bs L n_feat', L=L)
        return x

    def forward(self, data):
        # Unpack data
        strokes_seq = data['strokes_seq']
        canvas_seq = data['canvas_seq']
        strokes_ctx = data['strokes_ctx']
        canvas_ctx = data['canvas_ctx']
        imgs = data['img']

        bs = imgs.size(0)

        # Context
        img_feat = self.img_encoder(imgs)
        ctx_canvas_feat = self.encode_canvas(canvas_ctx)

        img_feat = torch.cat((img_feat, torch.zeros(bs, self.s_params, device=img_feat.device)), dim=1)
        ctx_sequence = torch.cat((ctx_canvas_feat, strokes_ctx), dim=-1)        # concatenate on features dim
        ctx_sequence = torch.cat((img_feat.unsqueeze(1), ctx_sequence), dim=1)  # concatenate on length dim
        ctx_sequence = self.proj_features(ctx_sequence)

        # Sequence
        x_canvas_feat = self.encode_canvas(canvas_seq)
        x_sequence = torch.cat((x_canvas_feat, strokes_seq), dim=-1)
        x_sequence = self.proj_features(x_sequence)


        # Permute sequences as length-first
        ctx_sequence = ctx_sequence.permute(1, 0, 2)
        x_sequence = x_sequence.permute(1, 0, 2)

        # Add positional encodings to the sequences
        ctx_sequence = self.context_PE(ctx_sequence)
        x_sequence = self.sequence_PE(x_sequence)

        return ctx_sequence, x_sequence
# ----------------------------------------------------------------------------------------------------------------------

class ContextEncoder(nn.Module):

    def __init__(self, config):
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
        if self.pool_context:
            self.context_token = nn.Parameter(torch.randn(1, 1, config["model"]["d_model"]))

    def forward(self, x):
        if self.pool_context:
            x = torch.cat((self.context_token, x), dim=0)   # add fake token
        x = self.net(x)

        if self.pool_context:
            return x[0]    # return only the fake token, summarizing the whole sequence
        else:
            return x       # retrun the whole sequence

# ----------------------------------------------------------------------------------------------------------------------

class TransformerVAE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = config["device"]
        self.s_params = config["model"]["n_strokes_params"]
        self.d_model = config["model"]["d_model"]
        self.seq_length = config["dataset"]["sequence_length"]
        self.context_length = config["dataset"]["context_length"]+1

        self.ctx_z = config["model"]["ctx_z"]   # how to merge context and z
        if self.ctx_z == 'proj':
            self.proj_ctx_z = nn.Linear(2 * self.d_model, self.d_model)

        self.time_queries_PE = PositionalEncoding(self.d_model,dropout=0)
        #self.query_dec = nn.Parameter(torch.randn(self.seq_length, self.d_model))
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
        if config["model"]["activation_last_layer"] == "sigmoid":
            self.prediction_head.add_module('act', nn.Sigmoid())


    def encode(self, x, context):
        bs = x.size(1)

        # add learnable tokens
        mu = repeat(self.mu, '1 1 dim -> 1 bs dim', bs=bs)
        log_sigma = repeat(self.log_sigma, '1 1 dim -> 1 bs dim', bs=bs)
        x = torch.cat((mu, log_sigma, x), dim=0)  # (T+2) x bs x d_model

        # Encode the input
        x = self.vae_encoder(x, context)
        mu = x[0]     # first element of the seq
        log_var = x[1]   # second element of the seq

        return mu, log_var

    def reparameterize(self, mu, log_sigma):

        sigma = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(sigma)

        z = eps.mul(sigma).add_(mu)
        return z

    def decode(self, size, z, context):
        time_queries = torch.zeros(size, device=self.device)
        time_queries = self.time_queries_PE(time_queries)

        if self.ctx_z == 'proj':
            # # Fuse z and context using projection
            z = repeat(z, 'bs n_feat -> ctx_len bs n_feat', ctx_len=self.context_length)
            z_ctx = torch.cat([context, z], dim=-1)   # concatenate on the feature dimension
            z_ctx = self.proj_ctx_z(z_ctx)
        elif self.ctx_z == 'cat':
            # Fuse z and context with concatenation on length dim
            z_ctx = torch.cat((context, z[None]), dim=0)
        else:
            raise NotImplementedError()

        out = self.vae_decoder(time_queries, z_ctx)

        # Linear proj
        out = out.permute(1, 0, 2)  # bs x L x dim
        out = self.prediction_head(out)

        return out

    def forward(self, seq, context):

        mu, log_sigma = self.encode(seq, context)
        z = self.reparameterize(mu, log_sigma)

        # Replicate z and decode
        out = self.decode(seq.size(), z, context)   # z is the input, context comes from the other branch

        return out, mu, log_sigma

    @torch.no_grad()
    def sample(self, L, ctx):
        # Sample z
        z = torch.randn(1, self.d_model, device=self.device)
        preds = self.decode(size=(L, 1, self.d_model), z=z, context=ctx)
        return preds

# ----------------------------------------------------------------------------------------------------------------------

class InteractivePainter(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.embedder = Embedder(config)
        self.context_encoder = ContextEncoder(config)
        self.transformer_vae = TransformerVAE(config)

    def forward(self, data):

        context, x = self.embedder(data)

        context_features = self.context_encoder(context)
        predictions, mu, log_sigma = self.transformer_vae(x, context_features)

        return predictions, mu, log_sigma

    @torch.no_grad()
    def generate(self, data, zero_context=False):
        context, x = self.embedder(data)
        context_features = self.context_encoder(context)
        if zero_context: # zero out the context to check if the model benefit from it
            context_features = torch.randn_like(context_features, device=context_features.device)
        predictions, mu, log_sigma = self.transformer_vae(x, context_features)
        return predictions


if __name__ == '__main__':

    from dataset_acquisition import StrokesDataset
    from torch.utils.data import DataLoader
    from utils.parse_config import ConfigParser
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default='a')
    parser.add_argument("--config", default='utils/config_local.yaml')
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()

    c_parser = ConfigParser(args)
    c_parser.parse_config(args)
    config = c_parser.get_config()


    dataset = StrokesDataset(config=config, split='train')

    dataloader = DataLoader(dataset, batch_size=2)
    data = next(iter(dataloader))

    # Define the model
    net = InteractivePainter(config)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    params = count_parameters(net)
    print(f'Number of trainable parameters: {params / 10**6}')
    preds, mu, l_sigma = net(data)


    gen = net.generate(data)

    labels = data['strokes_seq']
    criterion = torch.nn.MSELoss()
    loss = criterion(preds, labels)
    loss.backward()

    print(loss.item())
    print(preds.shape)