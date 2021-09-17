import torch
from torch.optim import AdamW, Adam
import torch.nn as nn
from model.utils.utils import AverageMeter, dict_to_device
import time
from model.training.losses import KLDivergence
import os
import datetime

class Trainer:

    def __init__(self, config, model, dataloader):

        # Optimizers
        self.checkpoint_path = config["train"]["checkpoint_path"]
        self.dataloader = dataloader
        self.optimizer = Adam(params=model.parameters(), lr=config["train"]["lr"], weight_decay=config["train"]['wd'])
        self.LRScheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.001, steps_per_epoch=len(dataloader), epochs=config["train"]["n_epochs"])

        # Losses
        self.MSELoss = nn.MSELoss()
        self.kl_lambda = config["train"]["kl_lambda"]
        self.KLDivergence = KLDivergence()

        # Misc
        self.device = config["device"]
        self.print_freq = config["train"]["print_freq"]

        if config["train"]["auto_resume"]["active"]:
            self.load_checkpoint(model, config["train"]["auto_resume"]["resume_path"])

    def save_checkpoint(self, model, filename=None):

        path = os.path.join(self.checkpoint_path, f"{filename}.pth.tar")
        torch.save({"model": model.state_dict(),
                    "optimizer": self.optimizer.state_dict()}, path)

        print(f'Model saved at {path}')

    def load_checkpoint(self, model, filename=None):
        ckpt = torch.load(filename, map_location=self.device)
        model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict((ckpt["optimizer"]))

        print(f'Model and optimizer loaded form {filename}')

    def train_one_epoch(self, model):
        # Set training mode
        model.train()

        mse_loss_meter = AverageMeter(name='mse_loss')
        kl_loss_meter = AverageMeter(name='kl')
        loss_meter = AverageMeter(name='tot_loss')
        batch_time = AverageMeter(name='batch_time')
        end = time.time()

        for idx, batch in enumerate(self.dataloader):
            batch = dict_to_device(batch, self.device)
            targets = batch['strokes_seq']

            predictions, mu, log_sigma = model(batch)
            mse_loss = self.MSELoss(predictions, targets)
            kl_div = self.KLDivergence(mu, log_sigma)

            loss = mse_loss + kl_div * self.kl_lambda

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            mse_loss_meter.update(mse_loss.item(), targets.size(0))
            kl_loss_meter.update(kl_div.item(), targets.size(0))
            loss_meter.update(loss.item(), targets.size(0))
            batch_time.update(time.time()-end)
            end = time.time()
            if idx % self.print_freq == 0:
                print(f'Iter : {idx} / {len(self.dataloader)}\t||\t'
                      f'Time : {str(datetime.timedelta(seconds=batch_time.val))} \t||\t'
                      f'MSE : {mse_loss_meter.val},  ({mse_loss_meter.avg})\t||\t'
                      f'KL : {kl_loss_meter.val}, ({kl_loss_meter.avg})')
            # Scheduler step
            self.LRScheduler.step()

        stats = {'mse' : mse_loss_meter.avg,
                 'kl' : kl_loss_meter.avg,
                 'loss' : loss_meter.avg}

        return stats