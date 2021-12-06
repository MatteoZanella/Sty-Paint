import torch
import math

def dict_to_device(inp, device, to_skip=[]) :
    return {k : t.to(device, non_blocking=True) for k, t in inp.items() if k not in to_skip}

class AverageMeter(object) :
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f') :
        self.name = name
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

    def __str__(self) :
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class LambdaScheduler :

    def __init__(self, config) :

        self.warmup_type = config["train"]["kl"]["warmup_type"]
        self.warmup_epochs = config["train"]["kl"]["warm_up_epochs"]
        self.base_value = config["train"]["kl"]["kl_lambda"]
        self.max_epochs = config["train"]["n_epochs"]

        assert self.warmup_type == 'linear' or self.warmup_type == 'cosine'
        if self.warmup_type == 'linear' :
            l1 = torch.linspace(0, self.base_value, self.warmup_epochs)
            l2 = torch.full((self.max_epochs - self.warmup_epochs,), self.base_value)
            self.schedule = torch.cat([l1, l2])
        elif self.warmup_type == 'cosine' :
            l1 = 0.5 * (0 - self.base_value) * (1 + torch.cos(
                torch.tensor(torch.arange(0, self.warmup_epochs) * math.pi / self.warmup_epochs))) + self.base_value
            l2 = torch.full((self.max_epochs - self.warmup_epochs,), self.base_value)
            self.schedule = torch.cat(([l1, l2]))

        assert len(self.schedule) == self.max_epochs
        # self.schedule.cuda()

    def __call__(self, ep) :
        return self.schedule[ep]