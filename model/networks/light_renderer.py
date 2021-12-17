import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL.Image as Image
import numpy as np

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

class Erosion2d(nn.Module):

    def __init__(self, m=1):
        super(Erosion2d, self).__init__()
        self.m = m
        self.pad = [m, m, m, m]
        self.unfold = nn.Unfold(2 * m + 1, padding=0, stride=1)

    def forward(self, x):
        batch_size, c, h, w = x.shape
        x_pad = F.pad(x, pad=self.pad, mode='constant', value=1e9)
        for i in range(c):
            channel = self.unfold(x_pad[:, [i], :, :])
            channel = torch.min(channel, dim=1, keepdim=True)[0]
            channel = channel.view([batch_size, 1, h, w])
            x[:, [i], :, :] = channel

        return x


class Dilation2d(nn.Module):

    def __init__(self, m=1):
        super(Dilation2d, self).__init__()
        self.m = m
        self.pad = [m, m, m, m]
        self.unfold = nn.Unfold(2 * m + 1, padding=0, stride=1)

    def forward(self, x):
        batch_size, c, h, w = x.shape
        x_pad = F.pad(x, pad=self.pad, mode='constant', value=-1e9)
        for i in range(c):
            channel = self.unfold(x_pad[:, [i], :, :])
            channel = torch.max(channel, dim=1, keepdim=True)[0]
            channel = channel.view([batch_size, 1, h, w])
            x[:, [i], :, :] = channel

        return x


class LightRenderer:
    def __init__(self, brushes_path, canvas_size):

        self.H = canvas_size
        self.W = canvas_size
        meta_brushes = self.load_meta_brushes(brushes_path)
        self.meta_brushes = meta_brushes.cuda()
        self.morph = False

    def load_meta_brushes(self, path):

        brush_large_vertical = read_img(path["large_vertical"], 'L', h=self.H, w=self.W)
        brush_large_horizontal = read_img(path["large_horizontal"], 'L', h=self.H, w=self.W)
        # brush_small_vertical = read_img(path["small_vertical"], 'L', h=self.H, w=self.W)
        # brush_small_horizontal = read_img(path["small_horizontal"], 'L', h=self.H, w=self.W)

        return torch.cat([brush_large_vertical, brush_large_horizontal], dim=0)

    def __call__(self, param):
        bs, L, dim = param.shape
        param = param.reshape(bs * L, dim)
        foregrounds, alphas = self.param2stroke(param)
        # foregrounds = foregrounds.reshape(bs, L, 3, self.H, self.W)
        # alphas = alphas.reshape(bs, L, 3, self.H, self.W)
        # valid_stroke = (valid_stroke * 1.0).reshape(bs, L)

        return foregrounds, alphas


    def render_all(self, param, canvas_start):
        bs, L, dim = param.shape
        param = param.reshape(bs * L, dim)
        foregrounds, alphas = self.param2stroke(param)

        # Morph
        if self.morph:
            foregrounds = Dilation2d(m=1)(foregrounds)
            alphas = Erosion2d(m=1)(alphas)

        # Reshape
        foregrounds = foregrounds.reshape(bs, L, 3, self.H, self.W)
        alphas = alphas.reshape(bs, L, 3, self.H, self.W)

        # Render all the strokes on the same canvas
        rec = canvas_start.clone()
        for j in range(foregrounds.shape[1]) :
            foreground = foregrounds[:, j, :, :, :]
            alpha = alphas[:, j, :, :, :]
            rec = foreground * alpha + rec * (1 - alpha)

        return rec

    def param2stroke(self, param):
        # param: b, 12
        b = param.shape[0]
        param_list = torch.split(param, 1, dim=1)
        x0, y0, w, h, theta = [item.squeeze(-1) for item in param_list[:5]]
        R0, G0, B0= param_list[5 :]
        #R2, G2, B2 = R0, G0, B0
        sin_theta = torch.sin(torch.acos(torch.tensor(-1., device=param.device)) * theta)
        cos_theta = torch.cos(torch.acos(torch.tensor(-1., device=param.device)) * theta)
        index = torch.full((b,), -1, device=param.device)

        # Add large / small brushstrokes templates
        # large_idx = (h * w) > 0.1
        # small_idx = (h * w) <= 0.1

        index[h > w] = 0
        index[h <= w] = 1
        # index[torch.logical_and(h > w, small_idx)] = 2
        # index[torch.logical_and(h <= w, small_idx)] = 3

        brush = self.meta_brushes[index.long()]
        alphas = torch.cat([brush, brush, brush], dim=1)
        alphas = (alphas > 0).float()
        # uncomment to interpolate between colors as in SNP
        #t = torch.arange(0, brush.shape[2], device=param.device).unsqueeze(0) / brush.shape[2]
        #color_map = torch.stack([R0 * (1 - t) + R2 * t, G0 * (1 - t) + G2 * t, B0 * (1 - t) + B2 * t], dim=1)
        # color_map = color_map.unsqueeze(-1).repeat(1, 1, 1, brush.shape[3])
        color_map = torch.stack([R0, G0, B0], dim=1)
        color_map = color_map.unsqueeze(-1).repeat(1, 1, brush.shape[2], brush.shape[3])

        brush = brush * color_map

        warp_00 = cos_theta / w
        warp_01 = sin_theta * self.H / (self.W * w)
        warp_02 = (1 - 2 * x0) * cos_theta / w + (1 - 2 * y0) * sin_theta * self.H / (self.W * w)
        warp_10 = -sin_theta * self.W / (self.H * h)
        warp_11 = cos_theta / h
        warp_12 = (1 - 2 * y0) * cos_theta / h - (1 - 2 * x0) * sin_theta * self.W / (self.H * h)
        warp_0 = torch.stack([warp_00, warp_01, warp_02], dim=1)
        warp_1 = torch.stack([warp_10, warp_11, warp_12], dim=1)
        warp = torch.stack([warp_0, warp_1], dim=1)
        grid = torch.nn.functional.affine_grid(warp, torch.Size((b, 3, self.H, self.W)), align_corners=False)
        brush = torch.nn.functional.grid_sample(brush, grid, align_corners=False)
        alphas = torch.nn.functional.grid_sample(alphas, grid, align_corners=False)

        # decision
        # decision = torch.logical_and(h != 0, w != 0)

        return brush, alphas