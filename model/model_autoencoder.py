import torch
import torch.nn as nn
from einops import rearrange, repeat
from model.networks.image_encoders import resnet18
from model.networks.layers import PE
from timm.models.layers import trunc_normal_


def count_parameters(net) :
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


class Embedder(nn.Module) :

    def __init__(self, config) :
        super(Embedder, self).__init__()

        self.s_params = config["model"]["n_strokes_params"]
        self.d_model = config["model"]["d_model"]
        self.context_length = config["dataset"]["context_length"]
        self.seq_length = config["dataset"]["sequence_length"]

        if config["model"]["pe"] == "sine" :
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

        # Permute sequences as length-first
        ctx_sequence = ctx_sequence.permute(1, 0, 2)

        return ctx_sequence


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

        if config["model"]["pe"] == 'sine' :
            self.time_queries_PE = PE(type='1d', d_model=self.d_model, length=self.seq_length)()
            self.time_queries_PE = self.time_queries_PE.permute(1, 0, 2)
        else :
            self.time_queries_PE = nn.Parameter(torch.zeros(self.seq_length, 1, self.d_model))
            trunc_normal_(self.time_queries_PE, std=0.02)

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


    def decode(self, context):

        time_queries = repeat(self.time_queries_PE.to(device=context.device), 'L 1 d_model -> L bs d_model', bs=context.size(1))
        out = self.vae_decoder(time_queries, context)

        # Linear proj
        out = out.permute(1, 0, 2)  # bs x L x dim
        out = self.prediction_head(out)

        return out

    def forward(self, context) :

        # Replicate z and decode
        out = self.decode(context)

        mu = torch.zeros((out.size(1), out.size(2)), device=context.device)
        log_sigma = torch.ones((out.size(1), out.size(2)), device=context.device)

        return out, mu, log_sigma


# ----------------------------------------------------------------------------------------------------------------------

class InteractivePainter(nn.Module) :

    def __init__(self, config) :
        super().__init__()

        self.embedder = Embedder(config)
        self.context_encoder = ContextEncoder(config)
        self.transformer_vae = TransformerVAE(config)

    def forward(self, data) :

        context = self.embedder(data)
        context_features = self.context_encoder(context)
        predictions, mu, log_sigma = self.transformer_vae(context_features)

        return predictions, mu, log_sigma

    @torch.no_grad()
    def generate(self, data, no_context=False, no_z=False) :
        context = self.embedder(data)
        context_features = self.context_encoder(context)
        if no_context or no_z:  # zero out the context to check if the model benefit from it
            context_features = torch.randn_like(context_features, device=context_features.device)

        predictions = self.transformer_vae(context_features)[0]

        return predictions


if __name__ == '__main__':
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

    data = {
        'strokes_ctx' : torch.randn((1,20, 12)),
        'strokes_seq' : torch.randn((1,8, 12)),
        'canvas' : torch.randn((1,3,256, 256)),
        'img' : torch.randn((1,3, 256, 256))
    }

    # Define the model
    net = InteractivePainter(config)

    def count_parameters(model) :
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    params = count_parameters(net)
    print(f'Number of trainable parameters: {params / 10 ** 6}')

    # Predict with context
    clean_preds = net(data)
