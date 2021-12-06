import torch.nn.functional as F
from einops import repeat, rearrange
import PIL.Image as Image
import numpy as np
import torch

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

class LightRenderer:
    def __init__(self, brushes_path, canvas_size):
        self.meta_brushes = self.load_meta_brushes(brushes_path)
        self.H = canvas_size
        self.W = canvas_size
        self.meta_brushes = F.interpolate(self.meta_brushes, (self.H, self.W))

    def load_meta_brushes(self, path):

        brush_large_horizontal = read_img(path.large_horizontal, 'L')
        brush_large_vertical = read_img(path.large_vertical, 'L')
        meta_brushes = torch.cat([brush_large_horizontal, brush_large_vertical], dim=0)
        return meta_brushes.cuda()

    def __call__(self, strokes):
        bs, L, _ = strokes.shape
        strokes = rearrange(strokes, 'bs L dim -> (bs L) dim')

        x0, y0 = strokes[:, 0], strokes[:, 1]
        w, h = strokes[:, 2], strokes[:, 3]
        theta = strokes[:, 4]
        R, G, B = strokes[:, 5], strokes[:, 6], strokes[:, 7]

        # Pre-compute sin theta and cos theta
        sin_theta = torch.sin(torch.acos(torch.tensor(-1.)) * theta)
        cos_theta = torch.cos(torch.acos(torch.tensor(-1.)) * theta)

        # to extend here
        index = torch.full((bs * L,), -1, dtype=torch.long)
        index[h > w] = 0
        index[h <= w] = 1
        brush = self.meta_brushes[index.long()]

        warp_00 = cos_theta / w
        warp_01 = sin_theta * self.H / (self.W * w)
        warp_02 = (1 - 2 * x0) * cos_theta / w + (1 - 2 * y0) * sin_theta * self.H / (self.W * w)
        warp_10 = -sin_theta * self.W / (self.H * h)
        warp_11 = cos_theta / h
        warp_12 = (1 - 2 * y0) * cos_theta / h - (1 - 2 * x0) * sin_theta * W / (H * h)
        warp_0 = torch.stack([warp_00, warp_01, warp_02], dim=1)
        warp_1 = torch.stack([warp_10, warp_11, warp_12], dim=1)
        warp = torch.stack([warp_0, warp_1], dim=1)
        # Conduct warping.
        grid = F.affine_grid(warp, [bs * L, 3, self.H, self.W], align_corners=False)
        brush = F.grid_sample(brush, grid, align_corners=False)

        alphas = (brush > 0).float()
        brush = brush.repeat(1, 3, 1, 1)
        #alphas = alphas.repeat(1, 3, 1, 1)

        color_map = torch.stack([R, G, B], dim=1)
        color_map = color_map.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.H, self.W)
        foreground = brush * color_map

        alphas = rearrange(alphas, '(bs L) c H W -> bs L c H W', bs=bs, L=L)
        foreground = rearrange(foreground, '(bs L) c H W -> bs L c H W', bs=bs, L=L)
        return foreground, alphas


    def render(self, strokes):
        assert strokes.size(0) == 1
        L = strokes.size(1)
        foreground, alphas = self(strokes=strokes)
        foreground = foreground[0].permute(0, 2, 3, 1).cpu().numpy()
        alphas = alphas[0].permute(0, 2, 3, 1).cpu().numpy()


        canvas = np.zeros((self.H, self.W, 3))

        for i in range(8) :
            canvas = canvas * (1 - alphas[i]) + foreground[i] * alphas[i]

        return np.uint8(canvas * 255.0)