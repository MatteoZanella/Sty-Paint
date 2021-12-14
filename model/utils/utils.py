import numpy as np

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
def cosine_scheduler(base_value, final_value, epochs, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs #* niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs
    return schedule