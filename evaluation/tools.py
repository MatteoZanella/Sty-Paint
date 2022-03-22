import numpy as np
import torch
from einops import repeat, rearrange
import cv2


def check_strokes(params, clamp_wh=1):
    if torch.is_tensor(params):
        params = torch.clamp(params, min=0, max=1)
        params[:, :, 2:4] = torch.clamp(params[:, :, 2:4], min=0.025, max=clamp_wh)
    else:
        params = np.clip(params, a_min=0, a_max=1)
        params[:, :, 2:4] = np.clip(params[:, :, 2:4], a_min=0.025, a_max=clamp_wh)
    return params


# ======================================================================================================================
# Visuals
def render_frames(params, batch, renderer):
    params = check_strokes(params)
    p_bs = params.shape[0]
    L = params.shape[1]

    c_bs = batch['canvas'].shape[0]
    n_samples = int(p_bs / c_bs)
    canvas_start = repeat(batch['canvas'], 'bs ch h w -> (bs n_samples) ch h w', n_samples=n_samples)

    frames = np.empty([p_bs, L, 256, 256, 3])
    alphas = np.empty([p_bs, L, 256, 256, 1])
    for i in range(p_bs):
        x = canvas_start[i].permute(1, 2, 0).cpu().numpy()
        for l in range(L):
            x, alpha = renderer.inference(params[i, l, :][None, None, :], canvas_start=x)
            frames[i, l] = x
            alphas[i, l] = alpha[0, :, :, None]

    return dict(frames=frames, alphas=alphas)


def dashed_cnt_pts(cnt, freq=2):
    on_off = np.concatenate((np.ones(freq), np.zeros(freq)))
    n_pts = cnt.shape[0]

    # On-off points
    on_off = np.tile(on_off, np.ceil(n_pts / (2 * freq)).astype('uint8'))
    on_off = on_off[:n_pts]

    return cnt[on_off == 1].squeeze()


def produce_visuals(params, renderer, starting_canvas, ctx=None, seq=None):
    params = check_strokes(params)
    red = (1, 0, 0)
    blue = (0, 0, 1)
    green = (0, 1, 0)

    final_result = np.copy(starting_canvas)
    if ctx is not None:
        final_result = renderer._render(ctx,
                                        canvas_start=final_result,
                                        highlight_border=True,
                                        color_border=blue)[0]

    if seq is not None:
        final_result, seq_alphas = renderer._render(seq,
                                                    canvas_start=final_result,
                                                    highlight_border=True,
                                                    color_border=green)

    final_result = renderer._render(params,
                                    canvas_start=final_result,
                                    highlight_border=True,
                                    color_border=red)[0]

    if seq is not None:
        for j in range(seq_alphas.shape[0]):
            cnt = cv2.findContours(seq_alphas[j, :, :, None].astype('uint8'),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)[0]
            pts = dashed_cnt_pts(cnt[0], freq=2)
            next = seq_alphas[j + 1:, :, :].sum(axis=0) > 0
            clean_pts = [pt for pt in pts if next[pt[1], pt[0]] == 0]
            clean_pts = np.stack(clean_pts)

            final_result[clean_pts[:, 1], clean_pts[:, 0]] = green

    return np.uint8(final_result * 255)


# ======================================================================================================================
# Features
def get_index(L, K=10):
    ids = []
    for i in range(L):
        for j in range(L):
            if (j > i + K) or (j < i - K) or i == j:
                continue
            else:
                ids.append([i, j])
    ids = np.array(ids)
    id0 = ids[:, 0]
    id1 = ids[:, 1]
    n = ids.shape[0]
    return id0, id1, n


def compute_features(x, ctx=None):
    if isinstance(x, list):
        x = np.concatenate(x, axis=0)
    if ctx is not None:
        if isinstance(ctx, list):
            ctx = np.concatenate(ctx, axis=0)

        x = np.concatenate((ctx, x), axis=1)

    _, L, n_params = x.shape
    id0, id1, n = get_index(L)
    dim_features = n * n_params

    bs = x.shape[0]
    if torch.is_tensor(x):
        feat = torch.empty((bs, dim_features), device=x.device)
    else:
        feat = np.empty((bs, dim_features))
    for j in range(n_params):
        feat[:, j * n: (j + 1) * n] = x[:, id0, j] - x[:, id1, j]

    feat_pos = feat[:, :2 * n]  # first two params are (x, y)
    feat_color = feat[:, 5 * n:]  # last 3 params are (r, g, b)
    return {"feat": feat, "feat_pos": feat_pos, "feat_color": feat_color}
