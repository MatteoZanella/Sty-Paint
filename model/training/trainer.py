import datetime
import time
import logging

import torch
from model.utils.utils import AverageMetersDict
from evaluation.metrics import FSD
import wandb


class Trainer:

    def __init__(self, config, model, train_dataloader, test_dataloader):

        self.config = config

        # Optimizers
        self.train_dataloader = train_dataloader
        self.n_iter_per_epoch = len(train_dataloader)
        self.test_dataloader = test_dataloader

        # Misc
        self.print_freq = config["train"]["logging"]["print_freq"]

        # set up model
        model.train_setup(n_iters_per_epoch=self.n_iter_per_epoch)

    def train_one_epoch(self, model, epoch):
        # Set training mode
        model.train()

        loss_avg_meters = AverageMetersDict(names=model.loss_names)
        info_avg_meters = AverageMetersDict(names=model.logs_names)

        start = time.time()
        for idx, batch in enumerate(self.train_dataloader):
            # batch = dict_to_device(batch)  # Already done in DataLoaderWrapper
            bs = batch['strokes_seq'].size(0)
            losses, loss_info = model.train_one_step(batch, epoch - 1, idx)
            del batch
            # update average meters
            loss_avg_meters.update(losses, bs)
            info_avg_meters.update(loss_info, bs)

            if idx % self.print_freq == 0:
                msg = f'Iter : {idx} / {self.n_iter_per_epoch}\t||\t'
                for name, val in loss_avg_meters.get_avg().items():
                    msg += f'{name} : {val:.3f} \t||\t'
                logging.info(msg)

        logging.info(f'EPOCH : {epoch} done! Time required : {str(datetime.timedelta(seconds=(time.time() - start)))} ')
        # Logging
        stats = dict()
        stats.update(loss_avg_meters.get_avg(header='train/'))
        stats.update(info_avg_meters.get_avg(header='train/'))
        stats.update({'train/epoch': epoch})

        return stats

    @torch.no_grad()
    def evaluate(self, model):

        model.eval()

        fsdMetric = FSD()
        original, ctx, predictions_z_random, predictions_z_enc = [], [], [], []

        avg_meters = AverageMetersDict(names=model.eval_metrics_names)
        log_meters = AverageMetersDict(names=model.eval_info)

        stats = {}
        for idx, batch in enumerate(self.test_dataloader):
            # data = dict_to_device(batch, to_skip=['strokes', 'time_steps'])  # Already done in DataLoaderWrapper
            targets = batch['strokes_seq']
            bs = targets.size(0)
            predictions, metrics, info, visual = model.test_one_step(batch, get_visual=idx == 0)

            # FSD
            original.append(batch['strokes_seq'].cpu().numpy())
            ctx.append(batch['strokes_ctx'].cpu().numpy())
            predictions_z_random.append(predictions["fake_data_random"].cpu().numpy())
            predictions_z_enc.append(predictions["fake_data_encoded"].cpu().numpy())

            del batch, predictions

            avg_meters.update(metrics, bs)
            log_meters.update(info, bs)

            if visual is not None:
                stats.update({
                    'generation_w_z': wandb.Image(visual['plot_w_z']),
                    'generation_wo_z': wandb.Image(visual['plot_wo_z'])
                })

        # logging
        msg = ''
        for name, val in avg_meters.get_avg().items():
            msg += f'{name} : {val:.3f} \t||\t'
        logging.info(msg)

        stats.update(avg_meters.get_avg(header='test/'))
        stats.update(log_meters.get_avg(header='test/'))

        # Compute FD from stored features
        fsd_z_random = fsdMetric(original=original, generated=predictions_z_random, ctx=ctx)
        stats.update({f'test/random_fd_{k}': v for k, v in fsd_z_random.items()})
        fsd_z_enc = fsdMetric(original=original, generated=predictions_z_enc, ctx=ctx)
        stats.update({f'test/random_fd_{k}': v for k, v in fsd_z_enc.items()})

        return stats
