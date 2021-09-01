import torch
import torch.nn as nn
from einops import rearrange, repeat
from networks import PositionalEncoding, ResNetEncoder
import importlib
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------

class ImageEmbedder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.s_params = config["model"]["n_strokes_params"]
        self.d_model = config["model"]["d_model"]
        self.seq_length = config["dataset"]["context_length"]


        self.enc_img = ResNetEncoder()
        self.enc_canvas = ResNetEncoder()


        self.encoder = nn.TransformerEncoder(
                                    encoder_layer=nn.TransformerEncoderLayer(
                                                  d_model=self.d_model,
                                                  nhead=config["model"]["encoder"]["n_heads"],
                                                  batch_first=True
                                    ),
                                    num_layers=2)

        # PE Encodings
        self.PE = PositionalEncoding(d_hid=self.d_model, n_position=self.seq_length)

    def forward(self, img, stroke_params, canvas):
        bs = stroke_params.size(0)

        # Encode image
        img_feat = self.enc_img(img)
        #img_feat = rearrange(img_feat, 'bs n_feat 1 1 -> bs 1 n_feat')
        img_feat = torch.cat([img_feat.unsqueeze(1), torch.zeros(bs, 1, self.s_params)], dim=-1)   # pad with zeros

        # Encode canvas
        canvas = rearrange(canvas, 'bs ctx c h w -> (bs ctx) c h w')
        canvas_feat = self.enc_canvas(canvas)
        canvas_feat = rearrange(canvas_feat, '(bs ctx) n_feat -> bs ctx n_feat', ctx=self.seq_length)

        ctx_feat = torch.cat([canvas_feat, stroke_params], dim=2)   # concatenate on the feature dimension
        ctx_feat = self.PE(ctx_feat)

        # Transformer Encoder
        feat = torch.cat([img_feat, ctx_feat], dim=1)   # concatenate on the sequence length dimension
        out = self.encoder(feat)
        return out

# ----------------------------------------------------------------------------------------------------------------------

class TransformerVAE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.s_params = config["model"]["n_strokes_params"]
        self.d_model = config["model"]["d_model"]
        self.seq_length = config["dataset"]["sequence_length"]

        self.enc_canvas = ResNetEncoder()

        # PE encoder
        self.PE = PositionalEncoding(d_hid=self.d_model,
                                     n_position=self.seq_length)

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
                            num_layers=config["model"]["vae_encoder"]["n_layers"])

        # Final projection head
        self.prediction_head = nn.Linear(self.d_model, self.s_params)


    def encode(self, x, context):
        bs = x.size(0)

        # add learnable tokens
        mu_x = torch.randn(bs, 1, self.d_model)
        log_var_x = torch.randn(bs, 1, self.d_model)
        x = torch.cat([mu_x, log_var_x, x], dim=1)  # batch_size x (T+2) x d_model

        # Encode the input
        x = self.encoder(x, context)
        mu = x[:, 0, :].unsqueeze(1)      # first element of the seq
        log_var = x[:, 1, :].unsqueeze(1)   # second element of the seq

        # Repeat the output
        mu = repeat(mu, 'bs 1 n_feat -> bs seq_len n_feat', seq_len=self.seq_length)
        log_var = repeat(log_var, 'bs 1 n_feat -> bs seq_len n_feat', seq_len=self.seq_length)

        return mu, log_var

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return eps * std + mu

    def decode(self, z, context):
        # Mix the latent vector with positional embeddings
        positional_encodings = self.PE(torch.zeros_like(z))
        z = self.pe_embedder(positional_encodings, z)

        # Decode
        out = self.decoder(z, context)
        return out

    def forward(self, stroke_params, canvas, ctx):
        # Encode canvas
        canvas = rearrange(canvas, 'bs seq c h w -> (bs seq) c h w')
        canvas_feat = self.enc_canvas(canvas)
        canvas_feat = rearrange(canvas_feat, '(bs seq) n_feat -> bs seq n_feat', seq=self.seq_length)

        # Concatenate storkes params and add positional emebddings
        seq_feat = torch.cat([canvas_feat, stroke_params], dim=2)  # concatenate on the feature dimension
        seq_feat = self.PE(seq_feat)

        mu, log_var = self.encode(seq_feat, ctx)
        z = self.reparametrize(mu, log_var)
        out = self.decode(z, ctx)   # z is the input, context comes from the other branch

        # Linear proj
        out = rearrange(out, 'bs seq_len n_feat -> (bs seq_len) n_feat')
        preds = self.prediction_head(out)
        preds = rearrange(preds, '(bs seq_len) s_params -> bs seq_len s_params', seq_len=self.seq_length)

        return preds


# ----------------------------------------------------------------------------------------------------------------------

class InteractivePainter(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.image_encoder = ImageEmbedder(config)
        self.transformer_vae = TransformerVAE(config)

    def forward(self, img_ref, strokes_ctx, strokes_seq, canvas_ctx, canvas_seq):

        ctx = self.image_encoder(img_ref, strokes_ctx, canvas_ctx)
        out = self.transformer_vae(strokes_seq, canvas_seq, ctx)

        return out



if __name__ == '__main__':

    from dataset import StrokesDataset
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    from parse_config import ConfigParser

    c_parser = ConfigParser('./config.yaml')
    c_parser.parse_config()
    config = c_parser.get_config()

    img_transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    c_transform = transforms.Compose([transforms.ToTensor()])
    dataset = StrokesDataset(config=config,
                             img_transform=img_transform,
                             canvas_transform=c_transform)

    dataloader = DataLoader(dataset, batch_size=2)
    ref_imgs, strokes_ctx, strokes_seq, canvas_ctx, canvas_seq = next(iter(dataloader))

    # Define the model
    net = InteractivePainter(config)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    params = count_parameters(net)
    print(f'Number of trainable parameters: {params / 10**6}')
    preds = net(ref_imgs, strokes_ctx, strokes_seq, canvas_ctx, canvas_seq)

    criterion = torch.nn.MSELoss()
    loss = criterion(preds, strokes_seq)
    print(loss.item())
    print(preds.shape)