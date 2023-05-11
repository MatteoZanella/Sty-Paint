import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_
from .image_encoders import resnet18, ConvEncoder, PatchEmbed
from .layers import PositionalEncoding
from .efdm import exact_feature_distribution_matching


class ContextEncoder(nn.Module):

    def __init__(self, config):
        super(ContextEncoder, self).__init__()

        self.s_params = config["model"]["n_strokes_params"]
        self.d_model = config["model"]["d_model"]
        self.context_length = config["dataset"]["context_length"]
        self.seq_length = config["dataset"]["sequence_length"]
        if "use_context" in config["model"]["context_encoder"]:
            self.use_context = config["model"]["context_encoder"]["use_context"]
        else:
            self.use_context = True
        # Stylization config
        self.use_style = "use_style" in config["model"]["context_encoder"] and config["model"]["context_encoder"]["use_style"]

        if self.use_style:
            self.use_style_efdm = config["model"]["context_encoder"]["use_style_efdm"]
            self.use_style_tokens = config["model"]["context_encoder"]["use_style_tokens"]
        else:
            self.use_style_tokens = False
            self.use_style_efdm = False

        self.PE = PositionalEncoding(config)

        self.visual_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        trunc_normal_(self.visual_token, std=0.02)
        if self.use_context:
            self.stroke_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
            trunc_normal_(self.stroke_token, std=0.02)
        if self.use_style_tokens:
            self.style_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
            trunc_normal_(self.style_token, std=0.02)

        if config["model"]["img_encoder"]["type"] == 'resnet18':
            self.img_encoder = resnet18(pretrained=True,
                                        layers_to_remove=['layer4', 'fc'])
            if self.use_context:
                self.canvas_encoder = resnet18(pretrained=True,
                                               layers_to_remove=['layer4', 'fc'])
            if self.use_style:
                self.style_encoder = resnet18(pretrained=True,
                                              layers_to_remove=['layer4', 'fc'])

        elif config["model"]["img_encoder"]["type"] == 'convenc':
            self.img_encoder = ConvEncoder(spatial_output_dim=config["model"]["img_encoder"]["visual_feat_hw"],
                                           features_dim=config["model"]["img_encoder"]["visual_feat_dim"])
            if self.use_context:
                self.canvas_encoder = ConvEncoder(spatial_output_dim=config["model"]["img_encoder"]["visual_feat_hw"],
                                                  features_dim=config["model"]["img_encoder"]["visual_feat_dim"])
            if self.use_style:
                self.style_encoder = ConvEncoder(spatial_output_dim=config["model"]["img_encoder"]["visual_feat_hw"],
                                                 features_dim=config["model"]["img_encoder"]["visual_feat_dim"])

        elif config["model"]["img_encoder"]["type"] == 'patchembed':
            self.img_encoder = PatchEmbed(img_size=config["dataset"]["resize"], patch_size=16,
                                          in_chans=3, embed_dim=config["model"]["img_encoder"]["visual_feat_dim"],
                                          norm_layer=None, flatten=False)
            if self.use_context:
                self.canvas_encoder = PatchEmbed(img_size=config["dataset"]["resize"], patch_size=16,
                                                 in_chans=3,
                                                 embed_dim=config["model"]["img_encoder"]["visual_feat_dim"],
                                                 norm_layer=None, flatten=False)
            if self.use_style:
                self.style_encoder = PatchEmbed(img_size=config["dataset"]["resize"], patch_size=16,
                                                in_chans=3,
                                                embed_dim=config["model"]["img_encoder"]["visual_feat_dim"],
                                                norm_layer=None, flatten=False)
        else:
            raise NotImplementedError("Encoder not available")

        if self.use_context:
            self.conv_proj = nn.Conv2d(in_channels=2 * config["model"]["img_encoder"]["visual_feat_dim"],
                                       out_channels=self.d_model,
                                       kernel_size=(1, 1))
            self.proj_features = nn.Linear(self.s_params, self.d_model)

        # Encoder sequence
        if "use_transformer" in config["model"]["context_encoder"]:
            self.use_transformer = config["model"]["context_encoder"]["use_transformer"]
        else:
            self.use_transformer = True
        if self.use_transformer:
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=config["model"]["d_model"],
                    nhead=config["model"]["context_encoder"]["n_heads"],
                    dim_feedforward=config["model"]["context_encoder"]["ff_dim"],
                    activation=config["model"]["context_encoder"]["act"],
                    dropout=config["model"]["dropout"]
                ),
                num_layers=config["model"]["context_encoder"]["n_layers"])

    def forward(self, data):
        ctx_sequence = []
        ## Visual context
        img, hres_img = self.img_encoder(data['img'])
        ## Style context
        if self.use_style:
            style, hres_style = self.style_encoder(data['style'])
            if self.use_style_tokens:
                style_feat = rearrange(style, 'bs ch h w -> (h w) bs ch')
                style_feat = style_feat + self.style_token
                ctx_sequence.append(style_feat)
            if self.use_style_efdm:
                img = exact_feature_distribution_matching(img, style)
                hres_img = exact_feature_distribution_matching(hres_img, hres_style)
        if self.use_context:
            # Encode Canvas
            canvas, hres_canvas = self.canvas_encoder(data['canvas'])
            # Concatenate image and canvas channels, then project
            visual_feat = self.conv_proj(torch.cat((img, canvas), dim=1))
            hres_visual_feat = torch.cat((hres_img, hres_canvas), dim=1)
        else:
            visual_feat = img
            hres_visual_feat = hres_img
        visual_feat = rearrange(visual_feat, 'bs ch h w -> (h w) bs ch')
        visual_feat = visual_feat + self.PE.pe_visual_tokens(device=visual_feat.device) + self.visual_token
        ctx_sequence.append(visual_feat)        
        ## Strokes context
        if self.use_context:
            strokes_ctx = rearrange(data['strokes_ctx'], 'bs L dim -> L bs dim')
            strokes_feat = self.proj_features(strokes_ctx)
            strokes_feat = strokes_feat + self.PE.pe_strokes_tokens(pos=strokes_ctx,
                                                                    device=strokes_feat.device) + self.stroke_token
            ctx_sequence.append(strokes_feat)

        # Merge Context
        ctx_sequence = torch.cat(ctx_sequence, dim=0)
        # Add PE
        if self.use_transformer:
            ctx_sequence = self.transformer_encoder(ctx_sequence)

        return ctx_sequence, hres_visual_feat
