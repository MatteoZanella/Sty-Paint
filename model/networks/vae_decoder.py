import torch
import torch.nn as nn
from einops import rearrange, repeat
from layers import PEWrapper, PositionalEncoding, positionalencoding1d
import torch.nn.functional as F

def get_act(name):
    if name == "sigmoid" :
        act = nn.Sigmoid()
    elif name == "relu" :
        act = nn.ReLU()
    elif name == "identity" :
        act = nn.Identity()
    else :
        raise NotImplementedError('Activation can be either: sigmoid, relu or identity')
    return act

class VAEDecoder(nn.Module) :
    def __init__(self, config) :
        super(VAEDecoder, self).__init__()

        self.device = config["device"]
        self.s_params = config["model"]["n_strokes_params"]
        self.d_model = config["model"]["d_model"]

        if config["model"]["encoder_pe"] == "new" :
            print('Using new encodings')
            self.PE = PositionalEncoding(config)
        else :
            self.PE = PEWrapper(config)

        self.ctx_z = config["model"]["ctx_z"]  # how to merge context and z
        if self.ctx_z == 'proj' :
            self.proj_ctx_z = nn.Linear(2 * self.d_model, self.d_model)

        # Divide the decoder in 2 modules
        self.position_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=config["model"]["vae_decoder"]["n_heads"],
                dim_feedforward=config["model"]["vae_decoder"]["ff_dim"],
                activation=config["model"]["vae_decoder"]["act"],
            ),
            num_layers=config["model"]["vae_decoder"]["n_layers"])


        act = get_act(config["model"]["activation_last_layer"])
        self.norm1 = nn.LayerNorm(self.d_model)
        self.position_head = nn.Sequential(
            nn.Linear(self.d_model, 2))
        self.position_head.add_module('act', act)

        # Color / Shape Decoder
        self.color_tokens_proj = nn.Linear(2 * config["model"]["img_encoder"]["hres_feat_dim"], self.d_model)
        self.color_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=config["model"]["vae_decoder"]["n_heads"],
                dim_feedforward=config["model"]["vae_decoder"]["ff_dim"],
                activation=config["model"]["vae_decoder"]["act"],
            ),
            num_layers=config["model"]["vae_decoder"]["n_layers"])

        act = get_act(config["model"]["activation_last_layer"])
        self.norm2 = nn.LayerNorm(self.d_model)
        self.color_head = nn.Sequential(
            nn.Linear(self.d_model, self.s_params-2))
        self.color_head.add_module('act', act)

    def bilinear_sampling_length_first(self, feat, pos):
        n_strokes = pos.size(0)
        feat_temp = repeat(feat, 'bs ch h w -> (L bs) ch h w', L = n_strokes)
        grid = rearrange(pos, 'L bs p -> (L bs) 1 1 p')

        pooled_features = F.grid_sample(feat_temp, 2 * grid - 1, align_corners=False, mode='bicubic')
        pooled_features = rearrange(pooled_features, '(L bs) ch 1 1 -> L bs ch', L=n_strokes)

        return pooled_features

    def forward(self, z,
                context,
                visual_features,
                L):

        # Concatenate z and context
        if self.ctx_z == 'proj':
            z = repeat(z, 'bs dim -> ctx_len bs dim', ctx_len=context.size(0))
            context = self.proj_ctx_z(torch.cat((context, z), dim=-1))  # cat on the channel dimension and project
        elif self.ctx_z == 'cat':
            context = torch.cat((context, z[None]), dim=0)              # cat on the length dimension

        # Positional encodings
        pos_tokens = positionalencoding1d(x=L, orig_channels=self.d_model)
        pos_tokens = repeat(pos_tokens, '1 L dim -> L bs dim', bs=context.size(1)).to(context.device)

        # Decode Position
        pos_tokens = self.position_decoder(pos_tokens, context)
        pos_pred = self.position_head(self.norm1(pos_tokens))  # L x bs x dim

        # Decode Shape / Color
        color_tokens = self.bilinear_sampling_length_first(visual_features, pos_pred)
        color_tokens = self.color_tokens_proj(color_tokens)
        color_tokens += self.PE.pe_strokes_tokens(pos=pos_pred, device=color_tokens.device)

        color_tokens = self.color_decoder(color_tokens, context)
        color_pred = self.color_head(self.norm2(color_tokens))

        # cat and return
        output = torch.cat((pos_pred, color_pred), dim=-1)
        output = rearrange(output, 'L bs dim -> bs L dim')

        return output