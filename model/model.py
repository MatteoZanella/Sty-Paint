import torch
import torch.nn as nn
from einops import rearrange, repeat
from networks.image_encoders import ResNetEncoder
from networks.layers import PositionalEncoding, SequenceMerger


class Embedder(nn.Module):

    def __init__(self, config):
        super(Embedder, self).__init__()

        self.s_params = config["model"]["n_strokes_params"]
        self.d_model = config["model"]["d_model"]
        self.context_length = config["dataset"]["context_length"]
        self.seq_length = config["dataset"]["sequence_length"]
        self.L = config["dataset"]["total_length"]


        # Networks
        self.img_encoder = ResNetEncoder()
        self.context_PE = PositionalEncoding(self.d_model, n_position=self.context_length)
        self.sequence_PE = PositionalEncoding(self.d_model, n_position=self.seq_length)

        # Merger
        self.SM = SequenceMerger(config)

    def encode_canvas(self, x):
        L = x.size(1)
        x = rearrange(x, 'bs L c h w -> (bs L) c h w')
        x = self.img_encoder(x)
        x = rearrange(x, '(bs L) n_feat -> bs L n_feat', L=L)
        return x

    def forward(self, data):
        # Unpack data
        x = data['sequence']
        context = data['context']
        imgs = data['ref_img']

        ## Context
        img_feat = self.img_encoder(imgs)
        # Encode canvas images
        ctx_canvas_feat = self.encode_canvas(context['canvas'])
        # Create context seq
        ctx_sequence = self.SM(strokes_params=context['strokes'],
                               canvas_feat=ctx_canvas_feat,
                               img_feat=img_feat)

        ## Sequence
        # Encode canvas images
        x_canvas_feat = self.encode_canvas(x['canvas'])
        # Create context seq
        x_sequence = self.SM(strokes_params=x['strokes'],
                               canvas_feat=x_canvas_feat,
                               img_feat=None)

        # Add positional encodings to the sequences
        ctx_sequence = self.context_PE(ctx_sequence, offset=1)
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
                                                  batch_first=True
                                    ),
                                    num_layers=config["model"]["encoder"]["n_layers"])

    def forward(self, x):
        x = self.net(x)
        return x

# ----------------------------------------------------------------------------------------------------------------------

class TransformerVAE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.s_params = config["model"]["n_strokes_params"]
        self.d_model = config["model"]["d_model"]
        self.seq_length = config["dataset"]["sequence_length"]
        self.context_length = config["dataset"]["context_length"]

        self.PE = PositionalEncoding(self.d_model, self.seq_length)

        #
        self.proj_z_context = nn.Linear(256 + 256, 256)

        # Define Encoder and Decoder
        self.vae_encoder = nn.TransformerDecoder(
                            decoder_layer=nn.TransformerDecoderLayer(
                                                                    d_model=self.d_model,
                                                                     nhead=config["model"]["vae_encoder"]["n_heads"],
                                                                     batch_first=True
                                                                     ),
                            num_layers=config["model"]["vae_encoder"]["n_layers"])

        self.vae_decoder = nn.TransformerDecoder(
                            decoder_layer=nn.TransformerDecoderLayer(
                                                                     d_model=self.d_model,
                                                                     nhead=config["model"]["vae_decoder"]["n_heads"],
                                                                     batch_first=True
                            ),
                            num_layers=config["model"]["vae_decoder"]["n_layers"])

        # Final projection head
        self.prediction_head = nn.Sequential(
            nn.Linear(self.d_model, self.s_params))


    def encode(self, x, context):
        bs = x.size(0)

        # add learnable tokens
        mu_x = torch.randn(bs, 1, self.d_model).to(x.device)
        log_var_x = torch.randn(bs, 1, self.d_model).to(x.device)
        x = torch.cat([mu_x, log_var_x, x], dim=1)  # batch_size x (T+2) x d_model

        # Encode the input
        x = self.vae_encoder(x, context)
        mu = x[:, 0, :]     # first element of the seq
        log_var = x[:, 1, :]   # second element of the seq

        return mu, log_var

    def sample_latent_z(self, mu, log_sigma):

        sigma = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(sigma)

        z = mu + (eps * sigma)
        return z

    def decode(self, size, z, context):
        positional_encodings = self.PE(torch.zeros(size)).to(z.device)

        # Fuse z and context
        z = repeat(z, 'bs n_feat -> bs ctx_len n_feat', ctx_len=self.context_length)
        z_ctx = torch.cat([context, z], dim=-1)
        z_ctx = self.proj_z_context(z_ctx)

        out = self.vae_decoder(positional_encodings, z_ctx)
        return out

    def forward(self, seq, context):

        mu, log_sigma = self.encode(seq, context)
        z = self.sample_latent_z(mu, log_sigma)

        # Replicate z and decode
        out = self.decode(seq.size(), z, context)   # z is the input, context comes from the other branch

        # Linear proj
        out = rearrange(out, 'bs seq_len n_feat -> (bs seq_len) n_feat')
        preds = self.prediction_head(out)
        preds = rearrange(preds, '(bs seq_len) s_params -> bs seq_len s_params', seq_len=self.seq_length)

        return preds, mu, log_sigma


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



if __name__ == '__main__':

    from dataset import StrokesDataset
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    from utils.parse_config import ConfigParser
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default='a')
    parser.add_argument("--config", default='utils/config_local.yaml')
    args = parser.parse_args()

    c_parser = ConfigParser(args)
    c_parser.parse_config()
    config = c_parser.get_config()


    dataset = StrokesDataset(config=config)

    dataloader = DataLoader(dataset, batch_size=2)
    data = next(iter(dataloader))

    # Define the model
    net = InteractivePainter(config)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    params = count_parameters(net)
    print(f'Number of trainable parameters: {params / 10**6}')
    preds, mu, l_sigma = net(data)


    labels = data['sequence']['strokes']
    criterion = torch.nn.MSELoss()
    loss = criterion(preds, labels)
    print(loss.item())
    print(preds.shape)