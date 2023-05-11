from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

def read_img(img_path, img_type='RGB', h=None, w=None):
    img = Image.open(img_path).convert(img_type)
    if h is not None and w is not None:
        img = img.resize((w, h))
    img = transforms.ToTensor()(img)

    return img

# @torch.jit.script
# def indirect_draw_canvas(canvas : torch.Tensor, brushes : torch.Tensor, colors : torch.Tensor, alphas : torch.Tensor):
#     for i in range(brushes.shape[0]):
#         colored_brush =  brushes[i] * colors[i]
#         canvas = colored_brush * alphas[i] + canvas * (1 - alphas[i])
#     return canvas

@torch.jit.script
def draw_canvas(brushes, colors, alphas, lengths : List[int]):
    # Divide the strokes into groups belonging to the same canvas
    brushes = brushes.split(lengths)
    colors = colors.split(lengths)
    alphas = alphas.split(lengths)
    # Create the canvases
    canvases = []
    canvases_alpha = []
    for b, c, a in zip(brushes, colors, alphas):
        b = b - (a.flip([0]).cumsum(dim=0).flip([0]) - a)
        b = b.clamp(0, 1)
        canvas = (c * b).sum(dim=0).half()
        canvas_alpha = a.sum(0).clamp(0, 1).half()
        canvases.append(canvas)
        canvases_alpha.append(canvas_alpha)
    return torch.stack(canvases), torch.stack(canvases_alpha)

class LightRenderer(nn.Module):
    def __init__(self, brush_paths, canvas_size, batch_size=4096):
        super().__init__()
        self.H = canvas_size
        self.W = canvas_size
        self.batch_size = batch_size
        self.register_buffer("meta_brushes", self.load_meta_brushes(brush_paths))

    def load_meta_brushes(self, paths):
        brush_large_vertical = read_img(paths["large_vertical"], 'L', h=self.H, w=self.W)
        brush_large_horizontal = read_img(paths["large_horizontal"], 'L', h=self.H, w=self.W)
        # Stack the brushes and convert to float16
        brushes = torch.stack([brush_large_vertical, brush_large_horizontal], dim=0).half()
        # whiten the brush
        return brushes

    def forward(self, strokes, lengths : Union[List[int], int], start_canvas=None):
        """
        Args:
            strokes: Tensor of shape [L, 8]. Each stroke is (x, y, w, h, angle, r, g, b).
            lengths: List of shape [bs] where elements are the number of strokes for each output canvas.
                     Integer corresponding to the number of strokes in each batch.
            start_canvas: Tensor of shape [bs, 3, H, W] 
        Returns:
            Tensor of shape [bs, 3, h, w].
        """
        if isinstance(lengths, int):
            lengths = [lengths] * (len(strokes) // lengths)
        # Create the brushes on the canvas
        if self.batch_size is None:
            colors, brushes = self.strokes2brushes(strokes)
        else:
            colors, brushes = self.batch_strokes2brushes(strokes)
        # Create the alpha channel
        alphas = brushes.where(brushes <= .6, torch.tensor(1, device=brushes.device, dtype=brushes.dtype))

        # ==== Direct computation ====
        canvas, canvas_alpha = draw_canvas(brushes, colors, alphas, lengths)
        if start_canvas is not None:
            start_canvas = F.interpolate(start_canvas, mode='bilinear', size=(self.H, self.W))
            canvas = canvas + start_canvas.half() * (1 - canvas_alpha)
        return canvas.float()
    
    def batch_strokes2brushes(self, strokes : torch.Tensor):
        colors, brushes = [], []
        for batch in strokes.split(self.batch_size):
            colors_batch, brushes_batch = self.strokes2brushes(batch)
            colors.append(colors_batch)
            brushes.append(brushes_batch)
        colors = torch.cat(colors, dim=0)
        brushes = torch.cat(brushes, dim=0)
        return colors, brushes

    def strokes2brushes(self, strokes):
        # strokes: [b, 12]
        L = strokes.shape[0]
        x0, y0, w, h, theta = strokes[:, :5].T
        colors = strokes[:, 5:8, None, None]
        # Meta brushes: [Large Vertical, Large Horizontal]
        brushes_idx = (h <= w).long()
        brushes = self.meta_brushes[brushes_idx]
        # ==== Affine transformation ====
        rad_theta = theta * torch.pi
        sin_theta = torch.sin(rad_theta)
        cos_theta = torch.cos(rad_theta)

        warp_00 = cos_theta / w
        warp_01 = sin_theta * self.H / (self.W * w)
        warp_02 = (1 - 2 * x0) * warp_00 + (1 - 2 * y0) * warp_01
        warp_10 = -sin_theta * self.W / (self.H * h)
        warp_11 = cos_theta / h
        warp_12 = (1 - 2 * y0) * warp_11 - (1 - 2 * x0) * -warp_10
        warp_0 = torch.stack([warp_00, warp_01, warp_02], dim=1)
        warp_1 = torch.stack([warp_10, warp_11, warp_12], dim=1)
        # Stack the affine transformation matrix and convert into float16
        warp = torch.stack([warp_0, warp_1], dim=1)
        # Convert the tensors a datatype accepted by grid_sample
        warp = warp.half() if brushes.is_cuda else warp.float()
        brushes = brushes.half() if brushes.is_cuda else brushes.float()
        # Apply the affine transformation
        grid = F.affine_grid(warp, (L, 1, self.H, self.W), align_corners=False)
        brushes = F.grid_sample(brushes, grid, align_corners=False)

        return colors, brushes