import os
import datetime
import time
import logging

import torch
from torch.optim import AdamW, Adam, SGD
import torch.nn as nn
from model.utils.utils import AverageMeter, dict_to_device, LambdaScheduler, render_save_strokes
from model.training.losses import KLDivergence, MSECalculator
from timm.scheduler.cosine_lr import CosineLRScheduler
from evaluation.metrics import FDMetricIncremental, maskedL2
from evaluation import tools as etools

from dataset_acquisition.decomposition.painter import Painter
from dataset_acquisition.decomposition.utils import load_painter_config
import wandb

class Trainer:

    def __init__(self, config, model, train_dataloader, test_dataloader):

        self.config = config
        self.model_type = config["model"]["model_type"]

        # Optimizers
        self.checkpoint_path = config["train"]["logging"]["checkpoint_path"]
        self.train_dataloader = train_dataloader
        print(config["train"]["optimizer"]['wd'])
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
            kl_div = AverageMeter(name='KL divergence'))
        if self.config["train"]["loss_weights"]["color_img"] > 0:
            loss_avg_meters.update({'mse_color_img' : AverageMeter(name='MSE color ref')})

        info_avg_meters = dict(
            batch_time = AverageMeter(name='Time of each batch'),
            grad_norm = AverageMeter(name='Grad Norm'),
            mu = AverageMeter(name='Mu'),
            sigma = AverageMeter(name='Sigma')
        )

        start = time.time()
        end = time.time()

        kl_lambda = self.kl_lambda_scheduler(ep-1)
        for idx, batch in enumerate(self.train_dataloader):
            batch = dict_to_device(batch, self.device)
            targets = batch['strokes_seq']

            predictions, mu, log_sigma = model(batch)

            # Compute Losses
            mse, losses_dict = self.MSE(predictions=predictions,
                                        targets=targets,
                                        ref_imgs = batch['img'])
            kl_div = self.KLDivergence(mu, log_sigma)

            loss = mse + kl_div * kl_lambda

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
            self.optimizer.step()
            # Gradient clipping
            self.LRScheduler.step_update(ep * self.n_iter_per_epoch + idx)

            # Update logging
            losses_dict.update({'kl_div' : kl_div})
            bs = targets.size(0)
            for loss_name in losses_dict.keys():
                loss_avg_meters[loss_name].update(losses_dict[loss_name], bs)

            info_avg_meters["mu"].update(torch.abs(mu).mean().data.item(), bs)
            info_avg_meters["sigma"].update(log_sigma.exp().mean().data.item(), bs)
            info_avg_meters["grad_norm"].update(grad_norm.item(), bs)
            info_avg_meters["batch_time"].update(time.time()-end)
            end = time.time()

            if idx % self.print_freq == 0:
                logging.info(f'Iter : {idx} / {self.n_iter_per_epoch}\t||\t'
                      f'Time : {str(datetime.timedelta(seconds=info_avg_meters["batch_time"].val))} \t||\t'
                      f'MSE : {loss_avg_meters["mse"].val},  ({loss_avg_meters["mse"].avg})\t||\t'
                      f'KL : {loss_avg_meters["kl_div"].val}, ({loss_avg_meters["kl_div"].avg})'
                      f'Grad Norm: {info_avg_meters["grad_norm"].val}, ({info_avg_meters["grad_norm"].avg})')

        logging.info(f'EPOCH : {ep} done! Time required : {str(datetime.timedelta(seconds=(time.time()-start)))} ')

        # Logging
        stats = dict()
        for loss_name, avg_meter in loss_avg_meters.items():
            stats.update({f'train/{loss_name}' : avg_meter.avg})
        stats.update({
                 'train/epoch' : ep,
                 'train/lr' : self.optimizer.param_groups[0]["lr"],
                 'train/kl_lambda' : kl_lambda,
                 'train/mu' : info_avg_meters["mu"].avg,
                 'train/sigma' : info_avg_meters["sigma"].avg})


        return stats

    @torch.no_grad()
    def evaluate(self, model, ep) :

        model.eval()

        loss_meters = dict(
            z = dict(
                    mse = AverageMeter(name='Total MSE'),
                    mse_position = AverageMeter(name='MSE position'),
                    mse_size = AverageMeter(name = 'MSE size'),
                    mse_theta = AverageMeter(name='MSE theta'),
                    mse_color = AverageMeter(name='MSE color'),
                    mse_color_img = AverageMeter(name='MSE color img')),
            test = dict(
                    mse = AverageMeter(name='Total MSE'),
                    mse_position = AverageMeter(name='MSE position'),
                    mse_size = AverageMeter(name = 'MSE size'),
                    mse_theta = AverageMeter(name='MSE theta'),
                    mse_color = AverageMeter(name='MSE color'),
                    mse_color_img = AverageMeter(name='MSE color img')))

        full_eval_flag = ep % 25 == 0
        if full_eval_flag:
            logging.info('Full evalutation active')

            z_maskedl2_avg_meter = AverageMeter(name='Masked l2 z')
            fd_calculator_z = FDMetricIncremental()

            test_maskedl2_avg_meter = AverageMeter(name='Masked l2 test')
            fd_calculator_test = FDMetricIncremental()

        logs = {}
        for idx, batch in enumerate(self.test_dataloader) :
            data = dict_to_device(batch, self.device, to_skip=['strokes', 'time_steps'])
            targets = data['strokes_seq']
            bs = targets.size(0)

            # Predict with context and z
            preds_z = model.module.generate(data, no_context=False, no_z=False)
            _, mse_with_z = self.MSE(preds_z, targets, data['img'], isEval=True)

            # Prediction without z, as at test time
            preds_test = model.module.generate(data, no_z=True, no_context=False)
            _, mse_without_z = self.MSE(preds_test, targets, data['img'], isEval=True)

            # Update average meters
            for loss_name, value in mse_with_z.items():
                loss_meters['z'][loss_name].update(value, bs)

            for loss_name, value in mse_without_z.items():
                loss_meters['test'][loss_name].update(value, bs)

            # Full eval
            if full_eval_flag:
                preds_z = etools.check_strokes(preds_z)
                preds_test = etools.check_strokes(preds_test)

                # Use z as during training
                visuals_z = etools.render_frames(params=preds_z.cpu().numpy(),
                                                 batch=batch,
                                                 renderer=self.pt)

                l2 = maskedL2(batch['img'], visuals_z['frames'], visuals_z['alphas'])
                z_maskedl2_avg_meter.update(l2.item(), bs)
                fd_calculator_z.update_queue(original=data['strokes_seq'], generated=preds_z)


                # No z, as at test time
                visuals_test = etools.render_frames(params=preds_test.cpu().numpy(),
                                                    batch=batch,
                                                    renderer=self.pt)
                l2 = maskedL2(batch['img'], visuals_test['frames'], visuals_test['alphas'])
                test_maskedl2_avg_meter.update(l2.item(), bs)
                fd_calculator_test.update_queue(original=data['strokes_seq'], generated=preds_test)

                if idx == 0:
                    logging.info('Rendering Images')
                    img_ref = etools.produce_visuals(params=batch['strokes_seq'],
                                                     batch=batch,
                                                     renderer=self.pt,
                                                     batch_id=0)
                    img_z = etools.produce_visuals(params=preds_z,
                                                   batch=batch,
                                                   renderer=self.pt,
                                                   batch_id=0)
                    img_test = etools.produce_visuals(params=preds_test,
                                                   batch=batch,
                                                   renderer=self.pt,
                                                   batch_id=0)

                    logs.update({
                        "media/img_ref" : wandb.Image(img_ref, caption=f"Reference Image"),
                        "media/img_z" : wandb.Image(img_z, caption=f"Img w z"),
                        "media/img_test" : wandb.Image(img_test, caption=f"Img w/o z")})


        logging.info(f'TEST: '
                     f'Clean MSE : {loss_meters["z"]["mse"].avg}\t||\t'
                     f'No z MSE : {loss_meters["test"]["mse"].avg}')


        for phase in loss_meters.keys():  # phase = z or = test
            for loss_name, avg_meter in loss_meters[phase].items():
                logs.update({
                    f'eval/{phase}/{loss_name}' : avg_meter.avg})

        if full_eval_flag:
            _, fd_z = fd_calculator_z.compute_fd()
            _, fd_test = fd_calculator_test.compute_fd()
            logs.update(
                {
                    'eval/z/fd_all' : fd_z['all'],
                    'eval/z/fd_color' : fd_z['color'],
                    'eval/z/fd_position' : fd_z['position'],
                    'eval/z/l2' : z_maskedl2_avg_meter.avg,

                    'eval/test/fd_all' : fd_test['all'],
                    'eval/test/fd_color' : fd_test['color'],
                    'eval/test/fd_position' : fd_test['position'],
                    'eval/test/l2' : test_maskedl2_avg_meter.avg
                })

        return logs