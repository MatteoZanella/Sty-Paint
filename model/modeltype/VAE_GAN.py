import os
import torch
import torch.nn as nn
from model.networks import context_encoder, encoder, decoder, discriminator
from torch.optim import AdamW
from timm.scheduler.cosine_lr import CosineLRScheduler
from model.training.losses import KLDivergence, ReconstructionLoss, ColorImageLoss, RenderImageLoss
from model.training.losses import GANLoss, cal_gradient_penalty
from evaluation.metrics import compute_color_difference
from evaluation.tools import produce_visuals
from model.utils.utils import cosine_scheduler
from dataset_acquisition.decomposition.painter import Painter
from dataset_acquisition.decomposition.utils import load_painter_config

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

class VAEGANModel(nn.Module) :

    def __init__(self, config):
        super(VAEGANModel, self).__init__()
        self.config = config
        self.context_encoder = context_encoder.ContextEncoder(config)
        self.vae_encoder = encoder.Encoder(config)
        self.vae_decoder = decoder.Decoder(config)
        self.renderer = Painter(args=load_painter_config(config["renderer"]["painter_config"]))

        # Disc
        self.netD = discriminator.Discriminator(config)

    def train_setup(self, n_iters_per_epoch):
        self.checkpoint_path = self.config["train"]["logging"]["checkpoint_path"]
        self.n_iters_per_epoch = n_iters_per_epoch

        self.netG_params = list(self.context_encoder.parameters()) + \
                                list(self.vae_encoder.parameters()) + \
                                list(self.vae_decoder.parameters())

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
                                            lr_min=self.config["train"]["optimizer"]["min_lr"] / self.config["model"]["discriminator"]["scale_lr"],
                                            warmup_lr_init=self.config["train"]["optimizer"]["warmup_lr"],
                                            warmup_t=int(self.config["train"]["optimizer"]["warmup_ep"] * self.n_iters_per_epoch),
                                            cycle_limit=1,
                                            t_in_epochs=False,
                                        )

        ## Set weights
        self.weights = dict(
            position=self.config["train"]["losses"]["reconstruction"]["weight"]["position"],
            size=self.config["train"]["losses"]["reconstruction"]["weight"]["size"],
            theta=self.config["train"]["losses"]["reconstruction"]["weight"]["theta"],
            color=self.config["train"]["losses"]["reconstruction"]["weight"]["color"],
            reference_img_color=self.config["train"]["losses"]["reference_img"]["color"]["weight"],
            random_reference_img_color=self.config["train"]["losses"]["reference_img"]["color_wo_z"]["weight"],
            reference_img_render=self.config["train"]["losses"]["reference_img"]["render"]["weight"],
            kl_div=cosine_scheduler(base_value=self.config["train"]["losses"]["kl"]["weight"],
                                    final_value=self.config["train"]["losses"]["kl"]["weight"],
                                    warmup_epochs=self.config["train"]["losses"]["kl"]["warmup_epochs"],
                                    epochs=self.config["train"]["n_epochs"],
                                    patience_epochs=self.config["train"]["losses"]["kl"]["patience_epochs"]),
            G=self.config["train"]["losses"]["gan"]["weight"]["G"],
            D=self.config["train"]["losses"]["gan"]["weight"]["D"])

        # Losses
        self.KLDivergence = KLDivergence()
        self.criterionRec = ReconstructionLoss(mode=self.config["train"]["losses"]["reconstruction"]["mode"])
        self.criterionColorImg = ColorImageLoss(config=self.config)
        self.criterionRefImg = RenderImageLoss(config=self.config)

        self.gan_mode = self.config["train"]["losses"]["gan"]["mode"]
        self.criterionGAN = GANLoss(gan_mode=self.gan_mode)

        # Additional info
        self.loss_names = ["enc_loss_position", "enc_loss_color", "enc_loss_size", "enc_loss_theta", "kl_div",
                           "enc_loss_reference_img_color", "enc_loss_reference_img_render",
                           "random_loss_reference_img_color", "random_loss_reference_img_render", "random_loss_G",
                           "random_loss_D", "enc_loss_G", "enc_loss_D"]

        self.logs_names = ["mu", "sigma", "kl_weight", "lrG", "grad_normG", "lrD", "grad_normD"]

        self.eval_metrics_names = ['random_loss_position', 'random_loss_size', 'random_loss_theta', 'random_loss_color',
                                   'random_loss_reference_img', 'random_color_l2', 'random_color_l1',
                                   'enc_loss_position', 'enc_loss_size', 'enc_loss_theta', 'enc_loss_color',
                                   'enc_loss_reference_img', 'enc_color_l2', 'enc_color_l1']
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


    def forward(self, batch, sample_z=True, seq_length=None):
        if seq_length is None:
            _, seq_length, _ = batch['strokes_seq'].shape

        # Encode z
        context, visual_features = self.context_encoder(batch)
        z, mu, log_sigma = self.vae_encoder(batch, context)

        fake_data_encoded = self.vae_decoder(z=z,
                                             context=context,
                                             visual_features=visual_features,
                                             seq_length=seq_length)
        fake_data_random = None
        if sample_z :
            random_z = torch.randn_like(z)
            fake_data_random = self.vae_decoder(z=random_z,
                                                context=context,
                                                visual_features=visual_features,
                                                seq_length=seq_length)

        out = dict(
            fake_data_encoded=fake_data_encoded,
            fake_data_random=fake_data_random,
            mu=mu,
            log_sigma=log_sigma,
            context = context
        )
        return out

    def step_D(self, real, fake, context):
        # fake
        pred_fake = self.netD(fake.detach(), context.detach())
        loss_D_fake = self.criterionGAN(pred_fake, target_is_real=False)

        # real
        pred_real = self.netD(real, context.detach())
        loss_D_real = self.criterionGAN(pred_real, target_is_real=True)
        loss_D = loss_D_fake + loss_D_real
        if self.gan_mode == 'wgangp':
            loss_D += cal_gradient_penalty(netD=self.netD,
                                           fake_data=fake.detach(),
                                           real_data=real,
                                           device=real.device)[0]
        return loss_D


    def train_one_step(self, batch, epoch, idx):
        # prediction = self.forward(batch, sample_z=self.sample_z)
        self.optimizerG.zero_grad()
        self.optimizerD.zero_grad()
        #  ==== Forward ====
        _, seq_length, _ = batch['strokes_seq'].shape
        # Encode z
        context, visual_features = self.context_encoder(batch)
        z, mu, log_sigma = self.vae_encoder(batch, context)
        fake_data_encoded = self.vae_decoder(z=z,
                                             context=context,
                                             visual_features=visual_features,
                                             seq_length=seq_length)
        #  ==== Compute Losses ====
        # 1 - kl divergence
        kl_div = self.KLDivergence(mu, log_sigma)

        # 2 - reconstruction loss
        loss_position, loss_size, loss_theta, loss_color = self.criterionRec(predictions=fake_data_encoded,
                                                        targets=batch["strokes_seq"])
        # 3 - reference image loss
        loss_reference_img_color = self.criterionColorImg(predictions=fake_data_encoded,
                                                        ref_imgs=batch['img'])
        loss_reference_img_render = self.criterionRefImg(predictions=fake_data_encoded,
                                                        ref_imgs=batch['img'],
                                                        canvas_start=batch['canvas'])

        # sum all the losses
        total_loss_G = loss_position * self.weights['position'] + \
                    loss_size * self.weights["size"] + \
                    loss_theta * self.weights["theta"] + \
                    loss_color * self.weights["color"] + \
                    loss_reference_img_color * self.weights["reference_img_color"] + \
                    loss_reference_img_render * self.weights["reference_img_render"] + \
                    kl_div * self.weights['kl_div'][epoch]

        total_loss_G.backward(retain_graph=True)
        #  ==== Sample z ====
        set_requires_grad([self.netD], False)

        random_z = torch.randn_like(z)
        fake_data_random = self.vae_decoder(z=random_z,
                                            context=context,
                                            visual_features=visual_features,
                                            seq_length=seq_length)

        random_loss_reference_img_color = self.criterionColorImg(predictions=fake_data_random,
                                                                 ref_imgs=batch['img'])
        gan_loss_G = self.criterionGAN(self.netD(fake_data_random, batch['strokes_ctx']), True)
        random_loss_G =  gan_loss_G * self.weights['G'] + \
                         random_loss_reference_img_color * self.weights['random_reference_img_color']
        random_loss_G.backward()

        # SGD step
        grad_normG = torch.nn.utils.clip_grad_norm_(self.netG_params,
                                                    self.config["train"]["optimizer"]["clip_grad"])
        self.optimizerG.step()
        self.LRSchedulerG.step_update(epoch * self.n_iters_per_epoch + idx)
        ## ========= Train Discriminator
        set_requires_grad([self.netD], True)

        random_loss_D = self.step_D(fake=fake_data_random,
                                    real=batch["strokes_seq"],
                                    context=batch['strokes_ctx'])

        total_loss_D = random_loss_D * self.weights['D']

        total_loss_D.backward()
        grad_normD = torch.nn.utils.clip_grad_norm_(self.netD.parameters(),
                                                    self.config["train"]["optimizer"]["clip_grad"])
        self.optimizerD.step()
        self.LRSchedulerD.step_update(epoch * self.n_iters_per_epoch + idx)


        # ===== LOG =====
        log_losses = dict(
            enc_loss_position = loss_position.item(),
            enc_loss_size = loss_size.item(),
            enc_loss_theta = loss_theta.item(),
            enc_loss_color= loss_color.item(),
            enc_loss_reference_img_color = loss_reference_img_color.item(),
            enc_loss_reference_img_render=loss_reference_img_render.item(),
            kl_div = kl_div.item(),
            random_loss_reference_img_color = random_loss_reference_img_color.item(),
            #enc_loss_G = enc_loss_G.item(),
            random_loss_G = random_loss_G.item(),
            #enc_loss_D = enc_loss_D.item(),
            random_loss_D = random_loss_D.item()
            )

        # additional info
        log_info = dict(
            mu=torch.abs(mu).mean().data.item(),
            sigma=log_sigma.exp().mean().data.item(),
            lrG=self.optimizerG.param_groups[0]["lr"],
            grad_normG=grad_normG.item(),
            kl_weight=self.weights['kl_div'][epoch],
            lrD=self.optimizerD.param_groups[0]["lr"],
            grad_normD=grad_normD.item())

        return log_losses, log_info

    @torch.no_grad()
    def test_one_step(self, batch, fd_z_random, fd_z_encoded, get_visual=False):
        # Forward
        predictions = self.forward(batch, sample_z=True)

        # Random z
        random_loss_position, random_loss_size, random_loss_theta, random_loss_color = self.criterionRec(predictions["fake_data_random"], batch['strokes_seq'])
        random_loss_reference_img = self.criterionColorImg(predictions=predictions["fake_data_random"],
                                                         ref_imgs=batch['img'])
        random_color_diff_l1, random_color_diff_l2 = compute_color_difference(predictions["fake_data_random"])

        # Encoded z
        enc_loss_position, enc_loss_size, enc_loss_theta, enc_loss_color = self.criterionRec(predictions["fake_data_encoded"],
                                                                                        batch['strokes_seq'])
        enc_loss_reference_img = self.criterionColorImg(predictions=predictions["fake_data_encoded"],
                                                      ref_imgs=batch['img'])
        enc_color_diff_l1, enc_color_diff_l2 = compute_color_difference(predictions["fake_data_encoded"])

        # Reference dataset
        ref_color_l1, ref_color_l2 = compute_color_difference(batch['strokes_seq'])

        # Fd
        fd_z_random.update_queue(original=batch["strokes_seq"], generated=predictions["fake_data_random"])
        fd_z_encoded.update_queue(original=batch["strokes_seq"], generated=predictions["fake_data_encoded"])

        visuals = None
        if get_visual :
            batch_id = 0
            plot_w_z = produce_visuals(predictions["fake_data_encoded"][batch_id].unsqueeze(0),
                                       ctx=batch["strokes_ctx"][batch_id].unsqueeze(0),
                                       renderer=self.renderer,
                                       st=batch['canvas'][0].permute(1, 2, 0).cpu().numpy(),
                                       seq=batch["strokes_seq"][batch_id].unsqueeze(0))
            plot_wo_z = produce_visuals(predictions["fake_data_random"][batch_id].unsqueeze(0),
                                        ctx=batch["strokes_ctx"][batch_id].unsqueeze(0),
                                        renderer=self.renderer,
                                        st=batch['canvas'][0].permute(1, 2, 0).cpu().numpy(),
                                        seq=batch["strokes_seq"][batch_id].unsqueeze(0))

            visuals = {'plot_w_z' : plot_w_z, 'plot_wo_z' : plot_wo_z}

        # logs
        eval_metrics = dict(
            random_loss_position = random_loss_position.item(),
            random_loss_size = random_loss_size.item(),
            random_loss_theta = random_loss_theta.item(),
            random_loss_color=random_loss_color.item(),
            random_loss_reference_img=random_loss_reference_img.item(),
            random_color_l2=random_color_diff_l2.item(),
            random_color_l1=random_color_diff_l1.item(),
            enc_loss_position=enc_loss_position.item(),
            enc_loss_size=enc_loss_size.item(),
            enc_loss_theta=enc_loss_theta.item(),
            enc_loss_color=enc_loss_color.item(),
            enc_loss_reference_img=enc_loss_reference_img.item(),
            enc_color_l2=enc_color_diff_l2.item(),
            enc_color_l1=enc_color_diff_l1.item()
        )

        logs_info = dict(
            ref_color_l1 = ref_color_l1.item(),
            ref_color_l2 = ref_color_l2.item()
        )

        return eval_metrics, logs_info, visuals