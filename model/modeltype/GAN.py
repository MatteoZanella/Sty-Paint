import os
import torch
import torch.nn as nn
from model.networks import context_encoder, decoder, discriminator
from torch.optim import AdamW
from timm.scheduler.cosine_lr import CosineLRScheduler
from model.training.losses import RenderImageLoss, ColorImageLoss
from model.training.losses import GANLoss, cal_gradient_penalty
from model.networks.light_renderer import LightRenderer
from evaluation.metrics import compute_color_difference

def set_requires_grad(nets, requires_grad=False) :
    """Set requires_grad=False for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list) :
        nets = [nets]
    for net in nets :
        if net is not None :
            for param in net.parameters() :
                param.requires_grad = requires_grad

class GANModel(nn.Module) :

    def __init__(self, config):
        super(GANModel, self).__init__()
        self.config = config

        # Generator
        self.context_encoder = context_encoder.ContextEncoder(config)
        self.decoder = decoder.Decoder(config)

        # Discriminator
        self.netD = discriminator.Discriminator(config)


    def train_setup(self, n_iters_per_epoch):
        self.checkpoint_path = self.config["train"]["logging"]["checkpoint_path"]
        self.n_iters_per_epoch = n_iters_per_epoch

        self.netG_params = list(self.context_encoder.parameters()) + \
                           list(self.decoder.parameters())

        self.optimizerG = AdamW(params=self.netG_params,
                               lr=self.config["train"]["optimizer"]["max_lr"],
                               weight_decay=self.config["train"]["optimizer"]['wd'],
                               betas=(self.config["train"]["optimizer"]["beta_1"],
                                      self.config["train"]["optimizer"]["beta_2"]))

        self.LRSchedulerG = CosineLRScheduler(
                                            self.optimizerG,
                                            t_initial=int(self.config["train"]["n_epochs"] * self.n_iters_per_epoch),
                                            t_mul=1.,
                                            lr_min=self.config["train"]["optimizer"]["min_lr"],
                                            warmup_lr_init=self.config["train"]["optimizer"]["warmup_lr"],
                                            warmup_t=int(self.config["train"]["optimizer"]["warmup_ep"] * self.n_iters_per_epoch),
                                            cycle_limit=1,
                                            t_in_epochs=False,
                                        )

        self.optimizerD = AdamW(params=self.netD.parameters(),
                                lr=self.config["train"]["optimizer"]["max_lr"] / self.config["model"]["discriminator"]["scale_lr"],
                                weight_decay=self.config["train"]["optimizer"]['wd'],
                                betas=(self.config["train"]["optimizer"]["beta_1"],
                                       self.config["train"]["optimizer"]["beta_2"]))

        self.LRSchedulerD = CosineLRScheduler(
                                            self.optimizerD,
                                            t_initial=int(self.config["train"]["n_epochs"] * self.n_iters_per_epoch),
                                            t_mul=1.,
                                            lr_min=self.config["train"]["optimizer"]["min_lr"]  / self.config["model"]["discriminator"]["scale_lr"],
                                            warmup_lr_init=self.config["train"]["optimizer"]["warmup_lr"],
                                            warmup_t=int(self.config["train"]["optimizer"]["warmup_ep"] * self.n_iters_per_epoch),
                                            cycle_limit=1,
                                            t_in_epochs=False,
                                        )

        ## set weights
        self.weights = dict(
            G = self.config["train"]["losses"]["gan"]["weight"]["G"],
            D = self.config["train"]["losses"]["gan"]["weight"]["D"],
            reference_img_color=self.config["train"]["losses"]["reference_img"]["color"]["weight"],
            reference_img_render=self.config["train"]["losses"]["reference_img"]["render"]["weight"],)

        # Losses
        self.criterionColorImg = ColorImageLoss(mode=self.config["train"]["losses"]["reference_img"]["mode"])
        self.criterionRefImg = RenderImageLoss(config = self.config)


        self.gan_mode = self.config["train"]["losses"]["gan"]["mode"]
        self.criterionGAN = GANLoss(gan_mode=self.gan_mode)

        # Additional info
        self.loss_names = ["random_loss_reference_img_color", "random_loss_reference_img_render", "random_loss_G",
                           "random_loss_D"]
        self.logs_names = ["lrG", "grad_normG", "lrD", "grad_normD", "p_real", "p_fake"]
        self.weights = {} # set by trainer

        self.eval_metrics_names = ['random_loss_reference_img', 'random_color_l2', 'random_color_l1']
        self.eval_info = ['ref_color_l1', 'ref_color_l2']

    def save_checkpoint(self, epoch, filename=None):
        if filename is None :
            path = os.path.join(self.checkpoint_path, f"checkpoint_{epoch}.pth.tar")
        else :
            path = os.path.join(self.checkpoint_path, f"latest.pth.tar")
        torch.save({"model" : self.state_dict(),
                    "optimizerG" : self.optimizerG.state_dict(),
                    "schedulerG" : self.LRSchedulerG.state_dict(),
                    "optimizerD" : self.optimizerD.state_dict(),
                    "schedulerD" : self.LRSchedulerD.state_dict(),
                    "epoch" : epoch,
                    "config" : self.config}, path)
        print(f'Model saved at {path}')


    def forward(self, batch, seq_length=None):
        if seq_length is None:
            _, seq_length, _ = batch['strokes_seq'].shape


        context, visual_features = self.context_encoder(batch)

        _, bs, dim = context.shape
        random_z = torch.randn((bs, dim), device=context.device)
        fake_data_random = self.decoder(z=random_z,
                                            context=context,
                                            visual_features=visual_features,
                                            seq_length=seq_length)

        out = dict(
            fake_data_random=fake_data_random,
            context = context
        )
        return out

    def step_D(self, real, fake, context) :
        # fake
        pred_fake = self.netD(fake.detach(), context.detach())
        loss_D_fake = self.criterionGAN(pred_fake, target_is_real=False)

        # real
        pred_real = self.netD(real, context.detach())
        loss_D_real = self.criterionGAN(pred_real, target_is_real=True)
        loss_D = loss_D_fake + loss_D_real

        # context
        if not self.config["model"]["discriminator"]["type"] == "transformer":
            context = None
        if self.gan_mode == 'wgangp' :
            loss_D += cal_gradient_penalty(netD=self.netD,
                                           fake_data=fake.detach(),
                                           real_data=real,
                                           context=context.detach(),
                                           device=real.device)[0]

        p_real = torch.sigmoid(pred_real)
        p_fake = torch.sigmoid(pred_fake)

        return loss_D, p_real, p_fake



    def train_one_step(self, batch, epoch, idx):

        prediction = self.forward(batch)

        ## ========= Train Discriminator
        set_requires_grad([self.netD], True)
        self.optimizerD.zero_grad()

        random_loss_D, p_real, p_fake  = self.step_D(fake=prediction["fake_data_random"],
                                                    real=batch["strokes_seq"],
                                                    context=prediction["context"])

        total_loss_D = random_loss_D * self.weights['D']

        total_loss_D.backward()
        grad_normD = torch.nn.utils.clip_grad_norm_(self.netD.parameters(),
                                                    self.config["train"]["optimizer"]["clip_grad"])
        self.optimizerD.step()
        self.LRSchedulerD.step_update(epoch * self.n_iters_per_epoch + idx)

        ## ========= Train Generator / Encoder
        self.optimizerG.zero_grad()

        # 1 - fool discriminator
        set_requires_grad([self.netD], False)
        random_loss_G = self.criterionGAN(self.netD(prediction["fake_data_random"], prediction["context"]), True)

        # 2 - reference image loss
        random_loss_reference_img_color = self.criterionColorImg(predictions=prediction["fake_data_random"],
                                                                ref_imgs=batch['img'])
        random_loss_reference_img_render = self.criterionRefImg(predictions=prediction["fake_data_random"],
                                                         ref_imgs=batch['img'],
                                                         canvas_start=batch['canvas'])

        # sum all the losses
        total_loss_G = random_loss_reference_img_color * self.weights["reference_img_color"] + \
                       random_loss_reference_img_render * self.weights["reference_img_render"] + \
                       random_loss_G * self.weights['G']

        total_loss_G.backward()
        grad_normG = torch.nn.utils.clip_grad_norm_(self.netG_params,
                                                    self.config["train"]["optimizer"]["clip_grad"])
        self.optimizerG.step()
        self.LRSchedulerG.step_update(epoch * self.n_iters_per_epoch + idx)

        # merge losses and loss info in two dicts
        log_losses = dict(
            random_loss_reference_img_color=random_loss_reference_img_color.item(),
            random_loss_reference_img_render=random_loss_reference_img_render.item(),
            random_loss_G=random_loss_G.item(),
            random_loss_D=random_loss_D.item())

        # additional info
        log_info = dict(
            lrG=self.optimizerG.param_groups[0]["lr"],
            grad_normG=grad_normG.item(),
            lrD=self.optimizerD.param_groups[0]["lr"],
            grad_normD=grad_normD.item(),
            p_real = p_real.detach().mean(),
            p_fake = p_fake.detach().mean())

        return log_losses, log_info

    @torch.no_grad()
    def test_one_step(self, batch, fd_z_random, fd_z_encoded):

        # Forward
        predictions = self.forward(batch)

        # Random z
        random_loss_reference_img = self.criterionRefImg(predictions=predictions["fake_data_random"],
                                                         ref_imgs=batch['img'],
                                                         canvas_start=batch['canvas'])
        random_color_diff_l1, random_color_diff_l2 = compute_color_difference(predictions["fake_data_random"])

        # Reference dataset
        ref_color_l1, ref_color_l2 = compute_color_difference(batch['strokes_seq'])

        # Fd
        fd_z_random.update_queue(original=batch["strokes_seq"], generated=predictions["fake_data_random"])

        # logs
        eval_metrics = dict(
            random_loss_reference_img=random_loss_reference_img.item(),
            random_color_l2=random_color_diff_l2.item(),
            random_color_l1=random_color_diff_l1.item(),
        )

        logs_info = dict(
            ref_color_l1 = ref_color_l1.item(),
            ref_color_l2 = ref_color_l2.item()
        )

        return eval_metrics, logs_info