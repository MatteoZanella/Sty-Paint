import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_
from model.networks.image_encoders import resnet18, ConvEncoder
from model.networks.layers import PEWrapper, PositionalEncoding
from model.networks.layers import positionalencoding1d


def count_parameters(net) :
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

class Embedder(nn.Module) :

    def __init__(self, config) :
        super(Embedder, self).__init__()

        self.s_params = config["model"]["n_strokes_params"]
        self.d_model = config["model"]["d_model"]
        self.context_length = config["dataset"]["context_length"]
        self.seq_length = config["dataset"]["sequence_length"]

        if config["model"]["encoder_pe"] == "new":
            print('Using new encodings')
            self.PE = PositionalEncoding(config)
        else:
            self.PE = PEWrapper(config)
        self.visual_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.stroke_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        trunc_normal_(self.visual_token, std=0.02)
        trunc_normal_(self.stroke_token, std=0.02)

        if config["model"]["img_encoder"]["type"] == 'resnet18':
            self.img_encoder = resnet18(pretrained=config["model"]["img_encoder"]["pretrained"],
                                        layers_to_remove=config["model"]["img_encoder"]["layers_to_remove"])
            self.canvas_encoder = resnet18(pretrained=config["model"]["img_encoder"]["pretrained"],
                                           layers_to_remove=config["model"]["img_encoder"]["layers_to_remove"])
        else:
            self.img_encoder = ConvEncoder(spatial_output_dim=config["model"]["img_encoder"]["visual_feat_hw"],
                                           features_dim=config["model"]["img_encoder"]["visual_feat_dim"])
            self.canvas_encoder = ConvEncoder(spatial_output_dim=config["model"]["img_encoder"]["visual_feat_hw"],
                                              features_dim=config["model"]["img_encoder"]["visual_feat_dim"])

        self.conv_proj = nn.Conv2d(in_channels=2 * config["model"]["img_encoder"]["visual_feat_dim"],
                                   out_channels=self.d_model,
                                   kernel_size=(1, 1))
        self.proj_features = nn.Linear(self.s_params, self.d_model)


    def forward(self, data) :
        strokes_seq = data['strokes_seq']
        strokes_ctx = data['strokes_ctx']
        img = data['img']
        canvas = data['canvas']

        # Encode Img/Canvas
        img, img_feat = self.img_encoder(img)
        canvas, canvas_feat = self.canvas_encoder(canvas)
        visual_feat = self.conv_proj(torch.cat((img, canvas), dim=1))
        hres_visual_feat = torch.cat((img_feat, canvas_feat), dim=1)

        # Everything as length first
        visual_feat = rearrange(visual_feat, 'bs ch h w -> (h w) bs ch')
        strokes_ctx = rearrange(strokes_ctx, 'bs L dim -> L bs dim')
        strokes_seq = rearrange(strokes_seq, 'bs L dim -> L bs dim')

        # Strokes
        ctx_sequence = self.proj_features(strokes_ctx)
        x_sequence = self.proj_features(strokes_seq)

        # Add PE
        visual_feat += self.PE.pe_visual_tokens(device=visual_feat.device)
        visual_feat += self.visual_token
        ctx_sequence += self.PE.pe_strokes_tokens(pos = strokes_ctx, device=ctx_sequence.device)
        ctx_sequence += self.stroke_token
        x_sequence += self.PE.pe_strokes_tokens(pos=strokes_seq, device=x_sequence.device)
        x_sequence += self.stroke_token

        # Merge Context
        ctx_sequence = torch.cat((visual_feat, ctx_sequence), dim=0)
        return ctx_sequence, x_sequence, hres_visual_feat

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
        self.width = config["dataset"]["resize"]
        if config["model"]["activation_last_layer"] == "sigmoid" :
            act = nn.Sigmoid()
        elif config["model"]["activation_last_layer"] == "relu":
            act = nn.ReLU()
        elif config["model"]["activation_last_layer"] == "identity":
            act = nn.Identity()
        else:
            raise NotImplementedError('Activation can be either: sigmoid, relu or identity')

        if config["model"]["encoder_pe"] == "new" :
            print('Using new encodings')
            self.PE = PositionalEncoding(config)
        else :
            self.PE = PEWrapper(config)

        self.ctx_z = config["model"]["ctx_z"]  # how to merge context and z
        if self.ctx_z == 'proj' :
            self.proj_ctx_z = nn.Linear(2 * self.d_model, self.d_model)

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
            ),
            num_layers=config["model"]["vae_encoder"]["n_layers"])

        # Divide the decoder in 2 modules
        self.position_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=config["model"]["vae_decoder"]["n_heads"],
                dim_feedforward=config["model"]["vae_decoder"]["ff_dim"],
                activation=config["model"]["vae_decoder"]["act"],
            ),
            num_layers=config["model"]["vae_decoder"]["n_layers"])

        self.position_head = nn.Sequential(
            nn.Linear(self.d_model, 2))
        self.position_head.add_module('act', act)

        # Color
        self.color_tokens_proj = nn.Linear(2 * config["model"]["img_encoder"]["hres_feat_dim"], self.d_model)
        self.color_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=config["model"]["vae_decoder"]["n_heads"],
                dim_feedforward=config["model"]["vae_decoder"]["ff_dim"],
                activation=config["model"]["vae_decoder"]["act"],
            ),
            num_layers=config["model"]["vae_decoder"]["n_layers"])

        self.color_head = nn.Sequential(
            nn.Linear(self.d_model, self.s_params-2))
        self.color_head.add_module('act', act)

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

    def bilinear_sampling_length_first(self, feat, pos):
        n_strokes = pos.size(0)
        feat_temp = repeat(feat, 'bs ch h w -> (L bs) ch h w', L = n_strokes)
        grid = rearrange(pos, 'L bs p -> (L bs) 1 1 p')

        pooled_features = F.grid_sample(feat_temp, 2 * grid - 1, align_corners=False, mode='bicubic')
        pooled_features = rearrange(pooled_features, '(L bs) ch 1 1 -> L bs ch', L=n_strokes)

        return pooled_features

    def decode(self, length, z, context, visual_features):
        bs = context.size(1)
        pos_tokens = positionalencoding1d(x=length, orig_channels=self.d_model)
        pos_tokens = repeat(pos_tokens, '1 L dim -> L bs dim', bs=bs).to(context.device)

        # Concatenate z and context
        if self.ctx_z == 'proj':
            z = repeat(z, 'bs dim -> ctx_len bs dim', ctx_len=context.size(0))
            context = self.proj_ctx_z(torch.cat((context, z), dim=-1))  # cat on the channel dimension and project
        elif self.ctx_z == 'cat':
            context = torch.cat((context, z[None]), dim=0)              # cat on the length dimension

        # Decode Position
        pos_tokens = self.position_decoder(pos_tokens, context)
        pos_pred = self.position_head(pos_tokens)  # L x bs x dim

        # Pool visual features with bilinear sampling
        color_tokens = self.bilinear_sampling_length_first(visual_features, pos_pred)
        color_tokens = self.color_tokens_proj(color_tokens)
        color_tokens += self.PE.pe_strokes_tokens(pos=pos_pred, device=color_tokens.device)

        color_tokens = self.color_decoder(color_tokens, context)
        color_pred = self.color_head(color_tokens)

        #
        output = torch.cat((pos_pred, color_pred), dim=-1)
        output = rearrange(output, 'L bs dim -> bs L dim')
        return output


    def forward(self, seq, context, visual_features) :

        mu, log_sigma = self.encode(seq, context)
        z = self.reparameterize(mu, log_sigma)

        # Decode
        out = self.decode(length=self.seq_length,
                          z = z,
                          context=context,
                          visual_features=visual_features)

        return out, mu, log_sigma

    @torch.no_grad()
    def sample(self, context, visual_features, L=None):
        if L is None:
            L = self.seq_length

        # Sample z
        bs = context.size(1)
        z = torch.randn(bs, self.d_model).to(self.device)
        context = torch.cat((context, z[None]), dim=0)


        preds = self.decode(length=L,
                            z = z,
                            context=context,
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
        self.eval()
        context, x, vs_features = self.embedder(data)
        context_features = self.context_encoder(context)

        if no_context :  # zero out the context to check if the model benefit from it
            context_features = torch.randn_like(context_features, device=context_features.device)

        if no_z :
            predictions = self.transformer_vae.sample(context=context_features, visual_features=vs_features)
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

    c_parser = ConfigParser(args.config)
    c_parser.parse_config(args)
    config = c_parser.get_config()

    # dataset = StrokesDataset(config=config, isTrain=True)

    # dataloader = DataLoader(dataset, batch_size=2)
    # data = next(iter(dataloader))

    data = {
        'strokes_ctx' : torch.rand((3, 10, 8)),
        'strokes_seq' : torch.rand((3, 8, 8)),
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

    gen = net.generate(data)

    loss = F.mse_loss(clean_preds, y)

    print(loss.item())
    loss.backward()