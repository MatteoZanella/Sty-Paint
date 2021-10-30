import numpy as np
import cv2
import torch

def render_frames(params, batch, renderer) :
    bs = params.shape[0]
    L = params.shape[1]
    frames = np.empty([bs, L, 256, 256, 3])
    alphas = np.empty([bs, L, 256, 256, 1])
    for i in range(bs) :
        x = batch['canvas'][i].permute(1, 2, 0).cpu().numpy()
        for l in range(L) :
            if (params[i, l, 2] <= 0.025) or (params[i, l, 3] <= 0.025) :
                params[i, l, 2] = 0.026
                params[i, l, 3] = 0.026

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

def check_strokes(params):
    params = torch.clamp(params, min=0, max=1)
    params[:, :, 2:4] = torch.clamp(params[:, :, 2:4], min=0.025, max=1)

    return params