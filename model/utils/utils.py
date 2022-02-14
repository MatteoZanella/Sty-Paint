import numpy as np
import cv2
import torchvision
from einops import rearrange, repeat
import torch.nn.functional as F

def dict_to_device(inp, to_skip=[]) :
    return {k : t.cuda(non_blocking=True) for k, t in inp.items() if k not in to_skip}

class AverageMeter(object) :
    """Computes and stores the average and current value"""

    def __init__(self, fmt=':f') :
        self.fmt = fmt
        self.reset()

    def reset(self) :
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1) :
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AverageMetersDict:
    def __init__(self, names):
        self.meters = dict()
        for name in names:
            self.meters.update({f'{name}': AverageMeter()})

    def update(self, vals, n):
        for k, v in vals.items():
            self.meters[k].update(v, n)

    def get_avg(self, header=''):
        output = dict()
        for k,v in self.meters.items():
            output.update({f'{header}{k}': v.avg})
        return output

    def get_val(self, key):
        return self.meters[key].val

#################################################################
def cosine_scheduler(base_value, final_value, epochs, warmup_epochs=0, start_warmup_value=0, patience_epochs=0):
    patience_schedule = np.array([])
    if patience_epochs > 0:
        patience_schedule = np.full((patience_epochs,), start_warmup_value)

    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs #* niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs - warmup_iters - patience_epochs)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((patience_schedule, warmup_schedule, schedule))
    assert len(schedule) == epochs
    return schedule

#######################################
def produce_visuals(params, ctx, renderer, st, seq=None) :
    fg, alpha = renderer.inference(params.cpu().numpy(), canvas_start=st)
    _, alpha_ctx = renderer.inference(ctx.cpu().numpy())
    cont = visualize(fg, alpha, alpha_ctx)
    if seq is not None:
        _, alpha_seq = renderer.inference(seq.cpu().numpy())
        alpha_seq = ((alpha_seq.sum(0) * 255)[:, :, None]).astype('uint8')
        contours_seq, _ = cv2.findContours(alpha_seq, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cont = cv2.drawContours(cont, contours_seq, -1, (0, 0, 255), 1)

    return cont


def visualize(foreground, alpha, alpha_ctx) :
    tmp = ((alpha.sum(0) * 255)[:, :, None]).astype('uint8')
    contours, hierarchy = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    tmp_alpha = ((alpha_ctx.sum(0) * 255)[:, :, None]).astype('uint8')
    contours_ctx, _ = cv2.findContours(tmp_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    x = (np.copy(foreground) * 255).astype('uint8')
    res = cv2.drawContours(x, contours_ctx, -1, (255, 0, 0), 1)
    res = cv2.drawContours(res, contours, -1, (0, 255, 0), 1)
    return res

######################################
def sample_color(pos, ref_imgs, blur=False):
    if blur:
        ref_imgs = torchvision.transforms.GaussianBlur((7,7))(ref_imgs)
    ref_imgs = repeat(ref_imgs, 'bs ch h w -> (bs L) ch h w', L=pos.size(1))
    grid = rearrange(pos, 'bs L dim -> (bs L) 1 1 dim')
    target_color_img = F.grid_sample(ref_imgs, 2 * grid - 1, align_corners=False, padding_mode='border')
    target_color_img = rearrange(target_color_img, '(bs L) ch 1 1 -> bs L ch', L=pos.size(1))
    return target_color_img