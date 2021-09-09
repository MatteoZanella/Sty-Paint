import torch
from torch.optim import AdamW, Adam
import torch.nn as nn
from model.utils.utils import AverageMeter, dict_to_device
import time
from model.training.losses import KLDivergence
import os

class Trainer:

    def __init__(self, config, model, dataloader, device):

        self.checkpoint_path = config["train"]["checkpoint_path"]
        self.kl_lambda = config["train"]["kl_lambda"]
        self.optimizer = Adam(params=model.parameters(), lr=config["train"]["lr"], weight_decay=config["train"]['wd'])
        self.dataloader = dataloader
        self.MSELoss = nn.MSELoss()
        self.KLDivergence = KLDivergence()
        self.device = device


    def save_checkpoint(self, model, filename=None):

        if filename is None:
            path = os.path.join(self.checkpoint_path, "latest.pth.tar")
        else:
            path = os.path.join(self.checkpoint_path, f"{filename}_.pth.tar")

        torch.save({"model": model.state_dict(),
                    "optimizer": self.optimizer.state_dict()}, path)

        print(f'Model saved at {path}')

    def load_checkpoint(self, model, filename=None):
        #TODO
        pass

    def train_one_epoch(self, model):

        # Set trainign mode
        model.train()

        mse_loss_meter = AverageMeter(name='mse_loss')
        kl_loss_meter = AverageMeter(name='kl')
        loss_meter = AverageMeter(name='tot_loss')
        batch_time = AverageMeter(name='batch_time')

        end = time.time()

        iter_mse_loss = []
        iter_kl_loss = []
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

            # Debug
            iter_mse_loss.append(mse_loss.item())
            iter_kl_loss.append(kl_div.item())

        print(loss_meter)
        print(mse_loss_meter)
        print(kl_loss_meter)
        print(batch_time)


        return iter_mse_loss, iter_kl_loss