import os
import datetime
import time
import logging

import torch
from torch.optim import AdamW, Adam, SGD
import torch.nn as nn
from model.utils.utils import AverageMeter, dict_to_device, LambdaScheduler, render_save_strokes
from model.training.losses import KLDivergence
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.cuda.amp import GradScaler

from dataset_acquisition.decomposition.painter import Painter
from dataset_acquisition.decomposition.utils import load_painter_config


class Trainer:

    def __init__(self, config, model, train_dataloader, test_dataloader):

        self.config = config
        self.model_type = config["model"]["model_type"]

        # Optimizers
        self.checkpoint_path = config["train"]["logging"]["checkpoint_path"]
        self.train_dataloader = train_dataloader
        print(config["train"]["optimizer"]['wd'])
        self.optimizer = AdamW(params=model.parameters(), lr=config["train"]["optimizer"]["max_lr"], weight_decay=config["train"]["optimizer"]['wd'])
        #self.LRScheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=config["train"]["optimizer"]["max_lr"], steps_per_epoch=len(dataloader), epochs=config["train"]["n_epochs"])
        self.n_iter_per_epoch = len(self.train_dataloader)
        self.LRScheduler = CosineLRScheduler(
                                            self.optimizer,
                                            t_initial=int(config["train"]["n_epochs"] * self.n_iter_per_epoch),
                                            t_mul=1.,
                                            lr_min=config["train"]["optimizer"]["min_lr"],
                                            warmup_lr_init=config["train"]["optimizer"]["warmup_lr"],
                                            warmup_t=int(config["train"]["optimizer"]["warmup_ep"] * self.n_iter_per_epoch),
                                            cycle_limit=1,
                                            t_in_epochs=False,
                                        )

        self.scaler = GradScaler()
        self.clip_grad = config["train"]["optimizer"]["clip_grad"]

        # Losses
        self.MSELoss = nn.MSELoss()
        self.kl_lambda_scheduler = LambdaScheduler(config)
        self.KLDivergence = KLDivergence()

        # Misc
        self.device = config["device"]
        self.print_freq = config["train"]["logging"]["print_freq"]

        if config["train"]["auto_resume"]["active"]:
            self.load_checkpoint(model, config["train"]["auto_resume"]["resume_path"])

        # Evaluation
        self.test_dataloader = test_dataloader
        self.checkpoint_path_render = config["train"]["logging"]["log_render_path"]
        self.pt = Painter(args=load_painter_config(config["render"]["painter_config"]))

    def save_checkpoint(self, model, epoch, filename=None):
        if filename is None:
            path = os.path.join(self.checkpoint_path, f"checkpoint_{epoch}.pth.tar")
        else:
            path = os.path.join(self.checkpoint_path, f"latest.pth.tar")

        if isinstance(model, nn.DataParallel):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        torch.save({"model": model_state_dict,
                    "optimizer": self.optimizer.state_dict(),
                    "lr_scheduler": self.LRScheduler.state_dict(),
                    "scaler" : self.scaler.state_dict(),
                    "epoch" : epoch,
                    "config" : self.config}, path),

        print(f'Model saved at {path}')

    def load_checkpoint(self, model, filename=None):
        ckpt = torch.load(filename, map_location=self.device)
        model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict((ckpt["optimizer"]))

        print(f'Model and optimizer loaded form {filename}')

    def train_one_epoch(self, model, ep):
        # Set training mode
        model.train()

        mse_loss_meter = AverageMeter(name='mse_loss')
        kl_loss_meter = AverageMeter(name='kl')
        #loss_meter = AverageMeter(name='tot_loss')
        batch_time = AverageMeter(name='batch_time')
        grad_norm_meter = AverageMeter(name='Gradient norm')
        mu_meter = AverageMeter(name='Mu')
        sigma_meter = AverageMeter(name='Sigma')
        start = time.time()
        end = time.time()

        kl_lambda = self.kl_lambda_scheduler(ep-1)
        for idx, batch in enumerate(self.train_dataloader):
            batch = dict_to_device(batch, self.device)
            targets = batch['strokes_seq']

            with torch.cuda.amp.autocast():
                predictions, mu, log_sigma = model(batch)
                mse_loss = self.MSELoss(predictions, targets)
                kl_div = self.KLDivergence(mu, log_sigma)

                #loss = mse_loss + kl_div * kl_lambda

            self.optimizer.zero_grad()
            #loss.backward()
            self.scaler.scale(mse_loss).backward(retain_graph=True)
            if self.model_type == 'autoencoder':
                kl_div = torch.tensor([0])
            else:
                self.scaler.scale(kl_div * kl_lambda).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.LRScheduler.step_update(ep * self.n_iter_per_epoch + idx)

            # Update logging
            bs = targets.size(0)
            mse_loss_meter.update(mse_loss.item(), bs)
            kl_loss_meter.update(kl_div.item(), bs)
            #loss_meter.update(torch.tensor(0).item(), bs)
            mu_meter.update(torch.abs(mu).mean().data.item(), bs)
            sigma_meter.update(log_sigma.exp().mean().data.item(), bs)
            grad_norm_meter.update(grad_norm.item(), bs)
            batch_time.update(time.time()-end)
            end = time.time()
            if idx % self.print_freq == 0:
                logging.info(f'Iter : {idx} / {self.n_iter_per_epoch}\t||\t'
                      f'Time : {str(datetime.timedelta(seconds=batch_time.val))} \t||\t'
                      f'MSE : {mse_loss_meter.val},  ({mse_loss_meter.avg})\t||\t'
                      f'KL : {kl_loss_meter.val}, ({kl_loss_meter.avg})'
                      f'Grad Norm: {grad_norm_meter.val}, ({grad_norm_meter.avg})')

        logging.info(f'EPOCH : {ep} done! Time required : {str(datetime.timedelta(seconds=(time.time()-start)))} ')

        # Logging
        stats = {'train/mse' : mse_loss_meter.avg,
                 'train/kl' : kl_loss_meter.avg,
                 #'loss' : loss_meter.avg,
                 'train/epoch' : ep,
                 'train/lr' : self.optimizer.param_groups[0]["lr"],
                 'train/kl_lambda' : kl_lambda,
                 'train/mu' : mu_meter.avg,
                 'train/sigma' : sigma_meter.avg}


        return stats

    @torch.no_grad()
    def evaluate(self, model, ep) :
        model.eval()
        mse_loss_meter = AverageMeter(name='mse_loss')
        mse_no_context_meter = AverageMeter(name='mse_without_context')
        mse_no_z_meter = AverageMeter(name='mse_no_z')

        logs = {}

        for idx, data in enumerate(self.test_dataloader) :
            data = dict_to_device(data, self.device, to_skip=['strokes', 'time_steps'])
            targets = data['strokes_seq']
            bs = targets.size(0)

            # Predict with context and z
            clean_preds = model.module.generate(data, no_context=False, no_z=False)
            clean_mse = self.MSELoss(clean_preds, targets)
            mse_loss_meter.update(clean_mse.item(), bs)

            # Predict without context
            noctx_preds = model.module.generate(data, no_context=True, no_z=False)
            noctx_mse = self.MSELoss(noctx_preds, targets)
            mse_no_context_meter.update(noctx_mse.item(), bs)

            # Prediction without z
            noz_preds = model.module.generate(data, no_z=True, no_context=False)
            noz_mse = self.MSELoss(noz_preds, targets)
            mse_no_z_meter.update(noz_mse.item(), bs)

            # Log some images
            if idx == 0 :
                # TOFIX: for now just plot the first element of the first batch
                imgs_to_log = render_save_strokes(generated_strokes=clean_preds[0][None],
                                                  original_strokes=targets[0][None],
                                                  painter=self.pt,
                                                  output_path=self.checkpoint_path_render,
                                                  ep=ep)
                logs.update(imgs_to_log)

        logging.info(f'TEST: '
                     f'Clean MSE : {mse_loss_meter.avg}\t||\t'
                     f'No ctx MSE : {mse_no_context_meter.avg}\t||\t'
                     f'No z MSE : {mse_no_z_meter.avg}')

        logs.update({'test/clean_mse' : mse_loss_meter.avg,
                     'test/no_context_mse' : mse_no_context_meter.avg,
                     'test/no_z_mse' : mse_no_z_meter.avg})

        return logs