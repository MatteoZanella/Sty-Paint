import os
import datetime
import time
import logging

import torch
from torch.optim import AdamW, Adam, SGD
import torch.nn as nn
from model.utils.utils import AverageMeter, dict_to_device, LambdaScheduler
from model.training.losses import KLDivergence, MSECalculator, GANLoss, ReferenceImageLoss
from timm.scheduler.cosine_lr import CosineLRScheduler
from evaluation.metrics import FDMetricIncremental, maskedL2
from evaluation import tools as etools

from model.networks.light_renderer import LightRenderer
import wandb

class Trainer:

    def __init__(self, config, model, train_dataloader, test_dataloader):

        self.config = config
        self.model_type = config["model"]["model_type"]

        # Optimizers
        self.checkpoint_path = config["train"]["logging"]["checkpoint_path"]
        self.train_dataloader = train_dataloader
        self.optimizer = AdamW(params=model.parameters(), lr=config["train"]["optimizer"]["max_lr"], weight_decay=config["train"]["optimizer"]['wd'])
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

        self.clip_grad = config["train"]["optimizer"]["clip_grad"]

        # Losses
        self.MSE = MSECalculator(loss_weights=config["train"]["loss_weights"])
        self.KLDivergence = KLDivergence()
        self.ContentLoss = ReferenceImageLoss(render=True, meta_brushes_path='' , canvas_size=256)

        # Weights
        self.kl_weight_schedule = LambdaScheduler(config) # use cosine anealing for KL weight, all the other are fixed

        # Misc
        self.print_freq = config["train"]["logging"]["print_freq"]

        # Evaluation
        self.test_dataloader = test_dataloader
        self.checkpoint_path_render = config["train"]["logging"]["log_render_path"]
        #self.renderer = LightRenderer()

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

        loss_avg_meters = dict(
            mse = AverageMeter(name='Total MSE'),
            mse_position = AverageMeter(name='MSE position'),
            mse_size = AverageMeter(name = 'MSE size'),
            mse_theta = AverageMeter(name='MSE theta'),
            mse_color = AverageMeter(name='MSE color'),
            kl_div = AverageMeter(name='KL divergence'),
            mse_color_img=AverageMeter(name='MSE color img'))

        info_avg_meters = dict(
            batch_time = AverageMeter(name='Time of each batch'),
            grad_norm = AverageMeter(name='Grad Norm'),
            mu = AverageMeter(name='Mu'),
            sigma = AverageMeter(name='Sigma')
        )

        start = time.time()
        end = time.time()

        kl_weight = self.kl_weight_schedule(ep-1)
        for idx, batch in enumerate(self.train_dataloader):
            batch = dict_to_device(batch, self.device)
            targets = batch['strokes_seq']

            predictions, mu, log_sigma = model(batch)

            # Compute Losses
            losses = {}

            losses.update(
                self.MSE(predictions=predictions, targets=targets))
            losses.update(
                self.ContentLoss(predictions=predictions, ref_imgs=batch['img']))
            losses.update(
                self.KLDivergence(mu, log_sigma))

            # Sum loss components
            total_loss = losses['mse_position'] * self.config["train"]["loss_weights"]["position"] + \
                losses['mse_size'] * self.config["train"]["loss_weights"]["size"] + \
                losses['mse_theta'] * self.config["train"]["loss_weights"]["theta"] + \
                losses['mse_color'] * self.config["train"]["loss_weights"]["color"] + \
                losses['mse_color_img'] * self.config["train"]["loss_weights"]["color_img"] + \
                losses['kl_div'] * kl_weight

            # Gradient step
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.LRScheduler.step_update(ep * self.n_iter_per_epoch + idx)

            # Update logging
            bs = targets.size(0)
            for loss_name in losses.keys():
                loss_avg_meters[loss_name].update(losses[loss_name], bs)

            info_avg_meters["mu"].update(torch.abs(mu).mean().data.item(), bs)
            info_avg_meters["sigma"].update(log_sigma.exp().mean().data.item(), bs)
            #info_avg_meters["grad_norm"].update(grad_norm.item(), bs)
            info_avg_meters["batch_time"].update(time.time()-end)
            end = time.time()

            if idx % self.print_freq == 0:
                logging.info(f'Iter : {idx} / {self.n_iter_per_epoch}\t||\t'
                      f'Time : {str(datetime.timedelta(seconds=info_avg_meters["batch_time"].val))} \t||\t'
                      #f'MSE : {loss_avg_meters["mse"].val},  ({loss_avg_meters["mse"].avg})\t||\t'
                      f'KL : {loss_avg_meters["kl_div"].val}, ({loss_avg_meters["kl_div"].avg})')
                      #f'Grad Norm: {info_avg_meters["grad_norm"].val}, ({info_avg_meters["grad_norm"].avg})')

        logging.info(f'EPOCH : {ep} done! Time required : {str(datetime.timedelta(seconds=(time.time()-start)))} ')

        # Logging
        stats = dict()
        for loss_name, avg_meter in loss_avg_meters.items():
            stats.update({f'train/{loss_name}' : avg_meter.avg})
        stats.update({
                 'train/epoch' : ep,
                 'train/lr' : self.optimizer.param_groups[0]["lr"],
                 'train/kl_lambda' : kl_weight,
                 'train/mu' : info_avg_meters["mu"].avg,
                 'train/sigma' : info_avg_meters["sigma"].avg})

        return stats

    @torch.no_grad()
    def evaluate(self, model) :

        model.eval()

        fd_test = FDMetricIncremental()
        fd_z = FDMetricIncremental()
        test_avg_meters = dict(
                            mse=AverageMeter(name='Total MSE'),
                            mse_position=AverageMeter(name='MSE position'),
                            mse_size=AverageMeter(name='MSE size'),
                            mse_theta=AverageMeter(name='MSE theta'),
                            mse_color=AverageMeter(name='MSE color'),
                            kl_div=AverageMeter(name='KL divergence'),
                            mse_color_img=AverageMeter(name='MSE color img'))

        z_avg_meters = dict(
                        mse = AverageMeter(name='Total MSE'),
                        mse_position = AverageMeter(name='MSE position'),
                        mse_size = AverageMeter(name = 'MSE size'),
                        mse_theta = AverageMeter(name='MSE theta'),
                        mse_color = AverageMeter(name='MSE color'),
                        kl_div = AverageMeter(name='KL divergence'),
                        mse_color_img=AverageMeter(name='MSE color img'))

        for idx, batch in enumerate(self.test_dataloader) :
            data = dict_to_device(batch, self.device, to_skip=['strokes', 'time_steps'])
            targets = data['strokes_seq']
            bs = targets.size(0)

            # Predict with context, sample z from gaussian
            preds_test = model.generate(data, use_z=False)
            test_losses = {}
            test_losses.update(self.MSE(preds_test, data['strokes_seq']))
            test_losses.update(self.ContentLoss(preds_test, data['img']))
            fd_test.update_queue(original=data['strokes_seq'], generated=preds_test)
            for name, val in test_losses.items():
                test_avg_meters[name].update(val.item(), bs)

            # Predict with z and context
            preds_z = model.generate(data, use_z=True)
            z_losses = {}
            z_losses.update(self.MSE(preds_z, data['strokes_seq']))
            z_losses.update(self.ContentLoss(preds_z, data['img']))
            fd_z.update_queue(original=data['strokes_seq'], generated=preds_z)
            for name, val in z_losses.items():
                z_avg_meters[name].update(val.item(), bs)

        # Compute FD
        _, fd_test = fd_test.compute_fd()
        _, fd_z = fd_z.compute_fd()

        test_avg_meters.update({f'fd_{k}' for k, v in fd_test.items()})
        z_avg_meters.update({f'fd_{k}' for k, v in fd_z.items()})

        stats = {}
        stats.update(
            {f'eval/test/{k}' for k, v in test_avg_meters.items()})
        stats.update(
            {f'eval/z/{k}' for k, v in z_avg_meters.items()}
        )
        return stats
