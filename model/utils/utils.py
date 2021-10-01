import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import wandb

def dict_to_device(inp, device, to_skip=[]):
    return {k : t.to(device) for k, t in inp.items() if k not in to_skip}

def render_save_strokes(generated_strokes, original_strokes, painter, output_path, ep):
    generated_strokes = generated_strokes.detach().cpu().numpy()

    # Check generated strokes
    checked_gen_strokes = painter.get_checked_strokes(generated_strokes)
    if len(checked_gen_strokes) == 0 :
        print('Skipping because of wrong format strokes')
        np.save(os.path.join(output_path, f'wrong_strokes_ep_{ep}.npy'), generated_strokes)
        return {}
    else :
        original, _ = painter.inference(strokes=original_strokes.cpu().numpy())
        generated, _ = painter.inference(strokes=generated_strokes)
        plt.imsave(os.path.join(output_path, f'original_{ep}.jpg'), original)
        plt.imsave(os.path.join(output_path, f'generated_{ep}.jpg'), generated)

    return {"generated": wandb.Image(generated, caption=f"generated_ep_{ep}"),
            "original": wandb.Image(original, caption=f"original_ep_{ep}")}

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class LambdaScheduler:

    def __init__(self, config):

        self.warmup_type = config["train"]["kl"]["warmup_type"]
        self.warmup_epochs = config["train"]["kl"]["warm_up_epochs"]
        self.base_value = config["train"]["kl"]["kl_lambda"]
        self.max_epochs = config["train"]["n_epochs"]

        assert self.warmup_type == 'linear' or self.warmup_type == 'cosine'
        if self.warmup_type == 'linear':
            l1 = torch.linspace(0, self.base_value, self.warmup_epochs)
            l2 = torch.full((self.max_epochs - self.warmup_epochs,), self.base_value)
            self.schedule = torch.cat([l1, l2])

        assert len(self.schedule) == self.max_epochs
        #self.schedule.cuda()

    def __call__(self, ep):
        return self.schedule[ep]