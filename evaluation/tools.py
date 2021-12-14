import numpy as np
import cv2
import torch
from einops import  repeat, rearrange
import torch.nn.functional as F

def render_frames(params, batch, renderer) :
    params = check_strokes(params)
    bs = params.shape[0]
    L = params.shape[1]
    frames = np.empty([bs, L, 256, 256, 3])
    alphas = np.empty([bs, L, 256, 256, 1])
    for i in range(bs) :
        x = batch['canvas'][i].permute(1, 2, 0).cpu().numpy()
        for l in range(L) :
            x, alpha = renderer.inference(params[i, l, :][None, None, :], canvas_start=x)
            frames[i, l] = x
            alphas[i, l] = alpha[0, :, :, None]

    return dict(frames=frames, alphas=alphas)

def draw_contours(foreground, alpha, alpha_ctx) :
    tmp = ((alpha.sum(0) * 255)[:, :, None]).astype('uint8')
    contours, hierarchy = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    tmp_alpha = ((alpha_ctx.sum(0) * 255)[:, :, None]).astype('uint8')
    contours_ctx, _ = cv2.findContours(tmp_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    x = (np.copy(foreground) * 255).astype('uint8')
    res = cv2.drawContours(x, contours_ctx, -1, (255, 0, 0), 1)
    res = cv2.drawContours(res, contours, -1, (0, 255, 0), 1)
    return res


def produce_visuals(params, batch, renderer, batch_id=0) :
    canvas_status = batch['canvas'][batch_id].permute(1,2,0).cpu().numpy()
    ctx = batch['strokes_ctx'][batch_id][None].cpu().numpy()

    # Render
    fg, alpha = renderer.inference(params.cpu().numpy(), canvas_start=canvas_status)
    _, alpha_ctx = renderer.inference(ctx)

    # Draw Contours
    cont = draw_contours(fg, alpha, alpha_ctx)
    return cont

def check_strokes(params, clamp_wh=1):
    if torch.is_tensor(params):
        params = torch.clamp(params, min=0, max=1)
        params[:, :, 2:4] = torch.clamp(params[:, :, 2:4], min=0.025, max=clamp_wh)
    else:
        params = np.clip(params, a_min=0, a_max=1)
        params[:, :, 2:4] = np.clip(params[:, :, 2:4], a_min=0.025, a_max=clamp_wh)
    return params

def render_lpips(inp, renderer, batch, bs, n_samples) :
    out = dict()
    for name in inp.keys() :
        out[name] = np.empty([bs, n_samples, 256, 256, 3])

        for b in range(bs) :
            for n in range(n_samples) :
                cs = batch['canvas'][b].permute(1, 2, 0).cpu().numpy()
                out[name][b, n :, :, :] = \
                renderer.inference(inp[name][n][b, :, :][None].cpu().numpy(), canvas_start=cs)[0]

    return out

def prepare_feature_difference(preds_lpips, bs, n_samples):
    out = dict()
    for key in preds_lpips.keys():
        out[key] = torch.empty((bs, n_samples, 8, 8))
        for n in range(n_samples) :
            out[key][:, n, :, :] = preds_lpips[key][n]

    return out


def sample_color(params, imgs) :
    if not torch.is_tensor(params):
        params = torch.tensor(params)
    bs, n_strokes, _ = params.shape
    img_temp = repeat(imgs, 'bs ch h w -> (bs L) ch h w', L=n_strokes)
    grid = rearrange(params[:, :, :2], 'bs L p -> (bs L) 1 1 p')
    color = F.grid_sample(img_temp, 2 * grid - 1, align_corners=False)
    color = rearrange(color, '(bs L) ch 1 1 -> bs L ch', L=n_strokes)
    #color = color.repeat(1,1,2)
    out_params = torch.cat((params.clone()[:, :, :5], color), dim=-1)
    return out_params.cpu().numpy()


def create_video(frames, path, size, scale=False):
    mul = 1 if not scale else 255
    video_writer = cv2.VideoWriter(
        path + '_animated.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20,
        (size, size))

    for this_frame in frames:
        video_writer.write((this_frame[:, :, : :-1] * mul).astype(np.uint8))