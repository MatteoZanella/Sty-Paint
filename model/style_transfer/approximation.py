from typing import List

import torch


@torch.jit.script
def inscribed_rects(params):
    """
    Given a rectangle of size (w, h) that has been rotated by 'angle' (in
    fraction of radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    Args:
        params: Tensor of shape [N, 3] where each row is (w, h, angle)
    Returns:
        Tensor of shape [N, 2] where each row is (wr, hr)
    """
    w, h, angle = params.T[0], params.T[1], params.T[2]

    w_longer = (w >= h)
    side_long = torch.where(w_longer, w, h)
    side_short = torch.where(w_longer, h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin, cos:
    rad_angle = angle * torch.pi
    sin_a = (rad_angle).sin().abs()
    cos_a = (rad_angle).cos().abs()

    # Boolean mask for the half constrained case
    h_c = (side_short <= 2. * sin_a * cos_a * side_long) | torch.isclose(sin_a, cos_a)

    wr = torch.zeros(params.shape[0], device=params.device)
    hr = torch.zeros(params.shape[0], device=params.device)

    #   Half constrained case: two crop corners touch the longer side,
    # the other two corners are on the mid-line parallel to the longer line
    x = .5 * side_short
    x_sin = x / sin_a
    x_cos = x / cos_a

    wr[h_c] = torch.where(w_longer[h_c], x_sin[h_c], x_cos[h_c])
    hr[h_c] = torch.where(w_longer[h_c], x_cos[h_c], x_sin[h_c])

    # Fully constrained case: crop touches all 4 sides
    f_c = ~h_c
    cos_2a = cos_a[f_c]**2 - sin_a[f_c]**2

    wr[~h_c] = (w[f_c] * cos_a[f_c] - h[f_c] * sin_a[f_c]) / cos_2a
    hr[~h_c] = (h[f_c] * cos_a[f_c] - w[f_c] * sin_a[f_c]) / cos_2a

    # Collapsed case
    collapsed = (w <= 0) | (h <= 0)
    wr[collapsed] = 0.
    hr[collapsed] = 0.

    bb = torch.stack([wr,hr], -1)
    
    return bb

@torch.jit.script
def approximate_strokes(strokes, rescaling : float = .75):
    """
    Approximate strokes with windows aligned to the axis. The windows are the maximum area rectangles
    that can be inscribed in the strokes.
    Args:
        strokes: Tensor of shape [L, 8]. Each stroke is (x, y, w, h, angle, r, g, b).
        rescaling: Rescale the strokes (w, h) by this factor. Compensate the master brushes paddings
    Returns:
        Tensor of shape [L, 4]. Each window is (x, y, wr, hr).
    """
    params = strokes[..., 2:5].clone().detach()  # Take [w, h, angle]
    params[..., :2] *= rescaling
    windows_sizes = inscribed_rects(params)
    windows = torch.cat([strokes[..., :2], windows_sizes], dim=-1)
    
    return windows

@torch.jit.script
def sample_windows(canvases, windows, lengths: List[int]):
    """
    Extract samples from canvas using the windows.
    Args:
        canvases: Tensor of shape [bs, 3, h, w]
        windows: Tensor of shape [L, 4]. Each window is (x, y, wr, hr).
        lengths: List of length [bs]. Each entry is the number of windows for the corresponding canvas.
    Returns:
        List containing [L] tensors of varying shape [3, hr, wr]
    """
    img_h, img_w = canvases.shape[-2:]
    # Transform the windows (x, y, wr, hr) to corners coordinates (x0, y0, x1, y1)
    x, y, wr, hr = windows.split(1, dim=-1)
    coords = torch.cat([x - wr / 2, y - hr / 2, x + wr / 2, y + hr / 2], dim=-1)
    coords.clamp_(0, 1)
    # Convert relative coordinates to pixel coordinates
    coords[:, [0, 2]] *= img_w
    coords[:, [1, 3]] *= img_h
    # Round to integers
    coords = coords.round_().int()
    # Set the second corner to be at least one pixel further from the first
    coords[:, 2:] += 1
    # Create the list of samples
    samples = []
    for canvas, group in zip(canvases, coords.split(lengths)):
        for p in group:
            x0, y0, x1, y1 = p[0], p[1], p[2], p[3]
            samples.append(canvas[:, y0:y1, x0:x1])
    
    return samples