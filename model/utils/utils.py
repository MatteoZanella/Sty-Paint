import numpy as np
import torch

from evaluation.tools import check_strokes

def dict_to_device(inp, to_skip=[], device=None) :
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return {k : t.to(device, non_blocking=True) if (torch.is_tensor(t) and k not in to_skip) else t for k, t in inp.items()}

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


def gram_matrix(x, normalize=True):
    """
    Compute the Gram matrix of x
    Args:
        x: (batch, C, ...)
    """
    features = x.flatten(2)
    (b, ch, n) = features.size()
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    return gram / (ch * n) if normalize else gram


def render_canvas(params, start_canvas, renderer):
    """
    Render the parameters on the starting canvases
    Args:
        params: (batch, L, 8)
        start_canvas: (batch, 3, H, W)
        renderer: Painter instance
    """
    # params = check_strokes(params)
    bs, L, n = params.shape
    if n == 8:
        # Add the second color and the alpha parameter
        params = torch.cat([params, params[:, :, -3:], torch.ones(bs, L, 1, device=params.device)], dim=2)

    images = renderer.neural_inference(params, start_canvas=start_canvas)

    return images