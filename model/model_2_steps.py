import torch
import torch.nn as nn
from einops import rearrange, repeat
from model.networks.image_encoders import resnet18
from model.networks.layers import PEWrapper
from timm.models.layers import trunc_normal_
import numpy as np

def count_parameters(net) :
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

def pos_2_idx(inp, visual_features_size=8):
    width = 256
    n = int(width / visual_features_size)
    with torch.no_grad():
        pos = rearrange(inp[:, :, :2], 'L bs dim -> bs L dim').detach().cpu().numpy()
        pos = np.rint(pos * (width - 1) + 0.5)

        x_i = (pos[:, :, 0] // n).astype('int')
        y_i = (pos[:, :, 1] // n).astype('int')

        flat_idx = y_i * visual_features_size + x_i
    return flat_idx

class Embedder(nn.Module) :

    def __init__(self, config) :
        super(Embedder, self).__init__()

        self.s_params = config["model"]["n_strokes_params"]
        self.d_model = config["model"]["d_model"]
        self.context_length = config["dataset"]["context_length"]
        self.seq_length = config["dataset"]["sequence_length"]
        self.visual_features_size = config["model"]["img_encoder"]["visual_features_dim"]

        self.PE = PEWrapper(config)
        self.visual_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.stroke_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        trunc_normal_(self.visual_token, std=0.02)
        trunc_normal_(self.stroke_token, std=0.02)

        self.img_encoder = resnet18(pretrained=config["model"]["img_encoder"]["pretrained"],
                                    layers_to_remove=config["model"]["img_encoder"]["layers_to_remove"])
        self.canvas_encoder = resnet18(pretrained=config["model"]["img_encoder"]["pretrained"],
                                       layers_to_remove=config["model"]["img_encoder"]["layers_to_remove"])

        self.conv_proj = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding=1, stride=1)
        self.conv_proj_ref = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1,1))
        self.proj_features = nn.Linear(self.s_params, self.d_model)

    def forward(self, data) :
        strokes_seq = data['strokes_seq']
        strokes_ctx = data['strokes_ctx']
        img = data['img']
        canvas = data['canvas']
        bs = img.shape[0]

        # Encode Img/Canvas
        img_feat, tmp_img = self.img_encoder(img)
        canvas_feat, tmp_canvas = self.canvas_encoder(canvas)

        # Visual Features to be use later
        visual_ref = torch.cat((tmp_img, tmp_canvas), dim=1)
        visual_ref = self.conv_proj_ref(visual_ref)
        visual_ref = rearrange(visual_ref, 'bs dim h w -> bs (h w) dim')

        # Visual Features used for context encoding
        visual_feat = self.conv_proj(torch.cat((img_feat, canvas_feat), dim=1))
        visual_feat = rearrange(visual_feat, 'bs dim h w -> bs h w dim')  # channels last

        # Strokes
        strokes = torch.cat((strokes_ctx, strokes_seq), dim=1)
        strokes_feat = self.proj_features(strokes)

        # Add 3D PE
        visual_feat, strokes_feat = self.PE(visual_feat, strokes_feat, strokes)
        visual_feat = rearrange(visual_feat, 'bs h w dim -> bs (h w) dim')

        visual_feat += self.visual_token
        strokes_feat += self.stroke_token

        # Rearrange
        strokes_ctx_feat = strokes_feat[:, :self.context_length, :]
        strokes_seq_feat = strokes_feat[:, self.context_length :, :]

        # Context
        ctx_sequence = torch.cat((visual_feat, strokes_ctx_feat), dim=1)

        # Length first
        ctx_sequence = rearrange(ctx_sequence, 'bs L dim -> L bs dim')
        x_sequence = rearrange(strokes_seq_feat, 'bs L dim -> L bs dim')

        return ctx_sequence, x_sequence, visual_ref

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
                dropout=config["model"]["encoder"]["dropout"]
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
        self.width = config["dataset"]["resize"]
        self.visual_features_size = config["model"]["img_encoder"]["visual_features_dim"]

        self.ctx_z = config["model"]["ctx_z"]  # how to merge context and z
        if self.ctx_z == 'proj' :
            self.proj_ctx_z = nn.Linear(2 * self.d_model, self.d_model)

        # Decoding tokens
        self.appearance_PE = nn.Parameter(torch.zeros(self.seq_length, 1, self.d_model))
        self.color_PE = nn.Parameter(torch.zeros(self.seq_length, 1, self.d_model))
        trunc_normal_(self.appearance_PE, std=0.02)
        trunc_normal_(self.color_PE, std=0.02)

        self.mu = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.log_sigma = nn.Parameter(torch.randn(1, 1, self.d_model))
        trunc_normal_(self.mu, std=0.02)
        trunc_normal_(self.log_sigma, std=0.02)

        # Define Encoder and Decoder
        self.vae_encoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=config["model"]["vae_encoder"]["n_heads"],
                dim_feedforward=config["model"]["vae_encoder"]["ff_dim"],
                activation=config["model"]["vae_encoder"]["act"],
                dropout=config["model"]["vae_encoder"]["dropout"]
            ),
            num_layers=config["model"]["vae_encoder"]["n_layers"])

        # Divide the decoder in 2 modules
        self.appearance_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=config["model"]["vae_decoder"]["n_heads"],
                dim_feedforward=config["model"]["vae_decoder"]["ff_dim"],
                activation=config["model"]["vae_decoder"]["act"],
                dropout=config["model"]["vae_decoder"]["dropout"]
            ),
            num_layers=config["model"]["vae_decoder"]["n_layers"])

        self.appearance_head = nn.Sequential(
            nn.Linear(self.d_model, 5),
            nn.Sigmoid())

        self.color_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=config["model"]["vae_decoder"]["n_heads"],
                dim_feedforward=config["model"]["vae_decoder"]["ff_dim"],
                activation=config["model"]["vae_decoder"]["act"],
                dropout=config["model"]["vae_decoder"]["dropout"]
            ),
            num_layers=config["model"]["vae_decoder"]["n_layers"])

        self.color_head = nn.Sequential(
            nn.Linear(self.d_model, 6),
            nn.Sigmoid())

    def encode(self, x, context):
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

    def decode(self, z, context, visual_features):
        bs = z.size(0)
        appearance_tokens = repeat(self.appearance_PE.to(device=z.device), 'L 1 d_model -> L bs d_model', bs=bs)
        z_ctx = torch.cat((context, z[None]), dim=0)

        # Decode Position
        appearance_tokens = self.appearance_decoder(appearance_tokens, z_ctx)
        #hidden_pos = rearrange(hidden_pos, 'L bs dim -> bs L dim')
        preds_appearance = self.appearance_head(appearance_tokens)

        # Pool visual features
        idxs = pos_2_idx(preds_appearance, visual_features_size=int(np.sqrt(visual_features.shape[1])))
        print(idxs)
        pooled_features = visual_features[np.arange(bs)[:, None], idxs]
        pooled_features = rearrange(pooled_features, 'bs L dim -> L bs dim')

        # Decode color
        color_tokens = repeat(self.color_PE.to(device=z.device), 'L 1 d_model -> L bs d_model', bs=bs)
        color_tokens = self.color_decoder(color_tokens, pooled_features)
        #hidden_color = rearrange(hidden_color, 'L bs dim -> bs L dim')
        preds_color = self.color_head(color_tokens)

        # Output
        out = torch.cat((preds_appearance, preds_color), dim=-1)
        out = rearrange(out, 'L bs dim -> bs L dim')

        return out

    def forward(self, seq, context, visual_features) :

        mu, log_sigma = self.encode(seq, context)
        z = self.reparameterize(mu, log_sigma)

        # Replicate z and decode
        out = self.decode(z=z,
                          context=context,
                          visual_features=visual_features)  # z is the input, context comes from the other branch

        return out, mu, log_sigma

    @torch.no_grad()
    def sample(self, ctx, visual_features):
        bs = ctx.size(1)
        z = torch.randn(bs, self.d_model).to(self.device)
        preds = self.decode(z=z,
                            context=ctx,
                            visual_features=visual_features)
        return preds


# ----------------------------------------------------------------------------------------------------------------------

class InteractivePainter(nn.Module) :

    def __init__(self, config):
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
    def generate(self, data, no_context=False, no_z=True) :
        context, x, vs_features = self.embedder(data)
        context_features = self.context_encoder(context)
        if no_context :  # zero out the context to check if the model benefit from it
            context_features = torch.randn_like(context_features, device=context_features.device)
        if no_z :
            predictions = self.transformer_vae.sample(ctx=context_features, visual_features=vs_features)
        else :
            predictions = self.transformer_vae(x, context_features, vs_features)[0]
        return predictions

if __name__ == '__main__' :
    import torch.nn.functional as F
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
        'strokes_ctx' : torch.randn((3, 10, 11)),
        'strokes_seq' : torch.randn((3, 8, 11)),
        'canvas' : torch.randn((3, 3, 256, 256)),
        'img' : torch.randn((3, 3, 256, 256))
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
    y = data['strokes_seq']
    clean_preds, _, _ = net(data)

    loss = F.mse_loss(clean_preds, y)

    print(loss.item())
    loss.backward()