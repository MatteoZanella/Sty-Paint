import os
import torch
import torch.nn.functional as F
import PIL.Image as Image
import numpy as np
from . import render_utils
import torch
import torch.nn as nn


class SignWithSigmoidGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        result = (x > 0).float()
        sigmoid_result = torch.sigmoid(x)
        ctx.save_for_backward(sigmoid_result)
        return result

    @staticmethod
    def backward(ctx, grad_result):
        (sigmoid_result,) = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = grad_result * sigmoid_result * (1 - sigmoid_result)
        else:
            grad_input = None
        return grad_input


class Painter(nn.Module):

    def __init__(self, param_per_stroke, total_strokes, hidden_dim, n_heads=8, n_enc_layers=3, n_dec_layers=3):
        super().__init__()
        self.enc_img = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, 3, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(True))
        self.enc_canvas = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, 3, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(True))
        self.conv = nn.Conv2d(128 * 2, hidden_dim, 1)
        self.transformer = nn.Transformer(hidden_dim, n_heads, n_enc_layers, n_dec_layers)
        self.linear_param = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, param_per_stroke))
        self.linear_decider = nn.Linear(hidden_dim, 1)
        self.query_pos = nn.Parameter(torch.rand(total_strokes, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(8, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(8, hidden_dim // 2))

    def forward(self, img, canvas):
        b, _, H, W = img.shape
        img_feat = self.enc_img(img)
        canvas_feat = self.enc_canvas(canvas)
        h, w = img_feat.shape[-2:]
        feat = torch.cat([img_feat, canvas_feat], dim=1)
        feat_conv = self.conv(feat)

        pos_embed = torch.cat([
            self.col_embed[:w].unsqueeze(0).contiguous().repeat(h, 1, 1),
            self.row_embed[:h].unsqueeze(1).contiguous().repeat(1, w, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        hidden_state = self.transformer(pos_embed + feat_conv.flatten(2).permute(2, 0, 1).contiguous(),
                                        self.query_pos.unsqueeze(1).contiguous().repeat(1, b, 1))
        hidden_state = hidden_state.permute(1, 0, 2).contiguous()
        param = self.linear_param(hidden_state)
        decision = self.linear_decider(hidden_state)
        return param, decision

def _normalize(x, width):
    return (int)(x * (width - 1) + 0.5)

def pad(img, H, W):
    b, c, h, w = img.shape
    pad_h = (H - h) // 2
    pad_w = (W - w) // 2
    remainder_h = (H - h) % 2
    remainder_w = (W - w) % 2
    img = torch.cat([torch.zeros((b, c, pad_h, w), device=img.device), img,
                     torch.zeros((b, c, pad_h + remainder_h, w), device=img.device)], dim=-2)
    img = torch.cat([torch.zeros((b, c, H, pad_w), device=img.device), img,
                     torch.zeros((b, c, H, pad_w + remainder_w), device=img.device)], dim=-1)
    return img

def read_img(img_path, img_type='RGB', h=None, w=None):
    img = Image.open(img_path).convert(img_type)
    if h is not None and w is not None:
        img = img.resize((w, h), resample=Image.NEAREST)
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float() / 255.
    return img


class PaintTransformer:

    def __init__(self, model_path, config, dev):
        self.stroke_num = 8
        self.input_size = config.canvas_size
        self.model_path = model_path
        self.patch_size = 32

        device = 'cpu'  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net_g = Painter(5, self.stroke_num, 256, 8, 3, 3).to(device)
        self.net_g.load_state_dict(torch.load(model_path))
        self.net_g.to(dev)
        self.net_g.eval()
        for param in self.net_g.parameters() :
            param.requires_grad = False


    def generate(self, data):
        bs = data['img'].shape[0]
        out = torch.empty([bs, 8, 11])
        for b in range(bs) :
            res = self.main(data['img'][b], data['canvas'][b],  data['strokes_ctx'][b])
            out[b] = res
        return out

    def get_ctx(self, ctx):
        # Location of the last stroke
        x_start, y_start = ctx[:, -1, :2][0]
        x_start = _normalize(x_start, self.input_size)
        y_start = _normalize(y_start, self.input_size)

        # Select window size based on average stroke area
        area = ctx[:, :, 2] * ctx[:, :, 3]   # h*w
        area = area.mean()
        if area < 0.004:
            windows_size = 32
        elif area < 0.01:
            windows_size = 64
        else:
            windows_size = 128
        print(f'Area: {area}, ws: {windows_size}')

        return (x_start, y_start), windows_size

    def main(self, original_img, canvas_start, strokes_ctx):

        patch_size = 32
        stroke_num = 8

        # Crop input
        st_point, ws = self.get_ctx(strokes_ctx[None])
        x1, x2, y1, y2 = render_utils.get_bbox(st_point, ws, self.input_size)

        # crop
        original = original_img[:, y1 :y2, x1 :x2][None]
        canvas_start = canvas_start[:, y1 :y2, x1 :x2][None]

        params, decision = self.predict(original, canvas_start)
        params = params.squeeze()

        params = torch.cat((params, params[:, -3:]), dim=-1)  # replicate the color, add a 0 for transparency, note that it won't be used
        params[:, 0] = (params[:, 0] * ws + x1) / self.input_size
        params[:, 1] = (params[:, 1] * ws + y1) / self.input_size
        params[:, 2] = (params[:, 2] * ws) / self.input_size
        params[:, 3] = (params[:, 3] * ws) / self.input_size

        return params

    def predict(self, original_img, canvas_start):

        patch_size = self.patch_size
        with torch.no_grad():
            #original_h, original_w = original_img.shape[-2:]
            K = 0
            #original_img_pad_size = patch_size * (2 ** K)
            #original_img_pad = pad(original_img, original_img_pad_size, original_img_pad_size)
            final_result = canvas_start
            layer_size = patch_size
            layer = 0
            img = F.interpolate(original_img, (layer_size, layer_size))
            result = F.interpolate(final_result, (patch_size * (2 ** layer), patch_size * (2 ** layer)))
            img_patch = F.unfold(img, (patch_size, patch_size), stride=(patch_size, patch_size))
            result_patch = F.unfold(result, (patch_size, patch_size),
                                    stride=(patch_size, patch_size))
            # There are patch_num * patch_num patches in total
            patch_num = (layer_size - patch_size) // patch_size + 1

            # img_patch, result_patch: b, 3 * output_size * output_size, h * w
            img_patch = img_patch.permute(0, 2, 1).contiguous().view(-1, 3, patch_size, patch_size).contiguous()
            result_patch = result_patch.permute(0, 2, 1).contiguous().view(
                -1, 3, patch_size, patch_size).contiguous()
            shape_param, stroke_decision = self.net_g(img_patch, result_patch)
            stroke_decision = SignWithSigmoidGrad.apply(stroke_decision)

            grid = shape_param[:, :, :2].view(img_patch.shape[0] * self.stroke_num, 1, 1, 2).contiguous()
            img_temp = img_patch.unsqueeze(1).contiguous().repeat(1, self.stroke_num, 1, 1, 1).view(
                img_patch.shape[0] * self.stroke_num, 3, patch_size, patch_size).contiguous()
            color = F.grid_sample(img_temp, 2 * grid - 1, align_corners=False).view(
                img_patch.shape[0], self.stroke_num, 3).contiguous()
            stroke_param = torch.cat([shape_param, color], dim=-1)
            # stroke_param: b * h * w, stroke_per_patch, param_per_stroke
            # stroke_decision: b * h * w, stroke_per_patch, 1
            param = stroke_param.view(1, patch_num, patch_num, self.stroke_num, 8).contiguous()
            decision = stroke_decision.view(1, patch_num, patch_num, self.stroke_num).contiguous().bool()
            # param: b, h, w, stroke_per_patch, 8
            # decision: b, h, w, stroke_per_patch
            param[..., :2] = param[..., :2] / 2 + 0.25
            param[..., 2:4] = param[..., 2:4] / 2

        return param, decision