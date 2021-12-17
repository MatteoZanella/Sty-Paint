import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_
from .image_encoders import resnet18, ConvEncoder, PatchEmbed
from .layers import PEWrapper, PositionalEncoding


class ContextEncoder(nn.Module) :

    def __init__(self, config) :
        super(ContextEncoder, self).__init__()

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
            self.img_encoder = resnet18(pretrained=True,
                                        layers_to_remove=['layer4', 'fc'])
            self.canvas_encoder = resnet18(pretrained=True,
                                           layers_to_remove=['layer4', 'fc'])
        elif config["model"]["img_encoder"]["type"] == 'convenc':
            self.img_encoder = ConvEncoder(spatial_output_dim=config["model"]["img_encoder"]["visual_feat_hw"],
                                           features_dim=config["model"]["img_encoder"]["visual_feat_dim"])
            self.canvas_encoder = ConvEncoder(spatial_output_dim=config["model"]["img_encoder"]["visual_feat_hw"],
                                              features_dim=config["model"]["img_encoder"]["visual_feat_dim"])

        elif config["model"]["img_encoder"]["type"] == 'patchembed':
            self.img_encoder = PatchEmbed(img_size=config["dataset"]["resize"], patch_size=16,
                                          in_chans=3, embed_dim=config["model"]["img_encoder"]["visual_feat_dim"],
                                          norm_layer=None, flatten=False)
            self.canvas_encoder = PatchEmbed(img_size=config["dataset"]["resize"], patch_size=16,
                                          in_chans=3, embed_dim=config["model"]["img_encoder"]["visual_feat_dim"],
                                          norm_layer=None, flatten=False)
        else:
            raise NotImplementedError("Encoder not available")

        self.conv_proj = nn.Conv2d(in_channels=2 * config["model"]["img_encoder"]["visual_feat_dim"],
                                   out_channels=self.d_model,
                                   kernel_size=(1, 1))
        self.proj_features = nn.Linear(self.s_params, self.d_model)


        # Encoder sequence
        self.transformer_encoder = nn.TransformerEncoder(
                                    encoder_layer=nn.TransformerEncoderLayer(
                                    d_model=config["model"]["d_model"],
                                    nhead=config["model"]["context_encoder"]["n_heads"],
                                    dim_feedforward=config["model"]["context_encoder"]["ff_dim"],
                                    activation=config["model"]["context_encoder"]["act"],
                                    dropout=config["model"]["dropout"]
                                    ),
                                    num_layers=config["model"]["context_encoder"]["n_layers"])


    def forward(self, data) :
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

        # Strokes
        ctx_sequence = self.proj_features(strokes_ctx)

        # Add PE
        visual_feat += self.PE.pe_visual_tokens(device=visual_feat.device)
        visual_feat += self.visual_token
        ctx_sequence += self.PE.pe_strokes_tokens(pos = strokes_ctx, device=ctx_sequence.device)
        ctx_sequence += self.stroke_token

        # Merge Context
        ctx_sequence = torch.cat((visual_feat, ctx_sequence), dim=0)

        # Transformer Encoder
        ctx_sequence = self.transformer_encoder(ctx_sequence)
        return ctx_sequence, hres_visual_feat