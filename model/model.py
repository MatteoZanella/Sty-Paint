import torch
import torch.nn as nn
from einops import rearrange, repeat
from networks.image_encoders import ResNetEncoder
from networks.layers import PositionalEncoding, Merger


class Embedder(nn.Module):

    def __init__(self, config, PE):
        super(Embedder, self).__init__()

        self.s_params = config["model"]["n_strokes_params"]
        self.d_model = config["model"]["d_model"]
        self.context_length = config["dataset"]["context_length"]
        self.seq_length = config["dataset"]["sequence_length"]
        self.L = config["dataset"]["total_length"]


        # Networks
        self.img_encoder = ResNetEncoder()
        self.canvas_encoder = ResNetEncoder()

        self.PE = PE

        # Merger
        self.merge = Merger(config)


    def forward(self, data):
        # Unpack data
        x = data['sequence']
        context = data['context']
        imgs = data['ref_img']

        # Encode reference images
        img_feat = self.img_encoder(imgs)
        img_feat = self.merge.pad_img_features(img_feat)

        # Concatenate all the canvas and compute the features
        canvas = torch.cat([context['canvas'], x['canvas']], dim=1)   # bs x (context_length+seq_length) x 3 x 512 x 512
        canvas = rearrange(canvas, 'bs L c h w -> (bs L) c h w')
        canvas_feat = self.canvas_encoder(canvas)
        canvas_feat = rearrange(canvas_feat, '(bs L) n_feat -> bs L n_feat', L=self.L)

        # Concatenate all the strokes
        strokes = torch.cat([context['strokes'], x['strokes']], dim=1) # bs x L x 12

        feat = self.merge.merge_strokes_canvas(strokes, canvas_feat)
        feat = self.PE(feat)

        # Split context form the sequence
        context = feat[:, :self.context_length, :]
        context = torch.cat([img_feat, context], dim=1)     # concatenate on seq dimension the reference images

        sequence = feat[:, self.context_length:, :]

        return context, sequence
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
    def __init__(self, config, PE):
        super().__init__()

        self.s_params = config["model"]["n_strokes_params"]
        self.d_model = config["model"]["d_model"]
        self.seq_length = config["dataset"]["sequence_length"]
        self.context_length = config["dataset"]["context_length"]


        self.PE = PE
        # Here is enough to compute cross attention probably
        self.pe_embedder = nn.TransformerDecoder(
                        decoder_layer=nn.TransformerDecoderLayer(
                                    d_model=self.d_model,
                                    nhead=config["model"]["vae_encoder"]["n_heads"],
                                    batch_first=True
                        ),
                        num_layers=1)

        # Define Encoder and Decoder
        self.encoder = nn.TransformerDecoder(
                            decoder_layer=nn.TransformerDecoderLayer(
                                                                    d_model=self.d_model,
                                                                     nhead=config["model"]["vae_encoder"]["n_heads"],
                                                                     batch_first=True
                                                                     ),
                            num_layers=config["model"]["vae_encoder"]["n_layers"])

        self.decoder = nn.TransformerDecoder(
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
        x = self.encoder(x, context)
        mu = x[:, 0, :]     # first element of the seq
        log_var = x[:, 1, :]   # second element of the seq

        return mu, log_var

    def sample_latent_z(self, mu, log_sigma):
        mu = repeat(mu, 'bs n_feat -> bs seq_len n_feat', seq_len=self.seq_length)
        log_sigma = repeat(log_sigma, 'bs n_feat -> bs seq_len n_feat', seq_len=self.seq_length)

        sigma = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(sigma)

        z = mu + (eps * sigma)
        return z

    def decode(self, z, context):
        # Mix the latent vector with positional embeddings
        positional_encodings = self.PE(torch.zeros_like(z), idx=self.context_length)
        z = self.pe_embedder(positional_encodings, z)

        # Decode
        out = self.decoder(z, context)
        return out

    def forward(self, seq, context):

        mu, log_sigma = self.encode(seq, context)
        z = self.sample_latent_z(mu, log_sigma)
        out = self.decode(z, context)   # z is the input, context comes from the other branch

        # Linear proj
        out = rearrange(out, 'bs seq_len n_feat -> (bs seq_len) n_feat')
        preds = self.prediction_head(out)
        preds = rearrange(preds, '(bs seq_len) s_params -> bs seq_len s_params', seq_len=self.seq_length)

        return preds, mu, log_sigma


# ----------------------------------------------------------------------------------------------------------------------

class InteractivePainter(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.PE = PositionalEncoding(d_hid=config["model"]["d_model"],
                                     n_position=config["dataset"]["total_length"])

        self.embedder = Embedder(config, self.PE)
        self.context_encoder = ContextEncoder(config)
        self.transformer_vae = TransformerVAE(config, self.PE)

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

    img_transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    c_transform = transforms.Compose([transforms.ToTensor()])
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