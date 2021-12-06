import torch
import torch.nn as nn
from model.networks import context_encoder, vae_encoder, vae_decoder, discriminator


def reparameterize(mu, log_sigma) :
    sigma = torch.exp(0.5 * log_sigma)
    eps = torch.randn_like(sigma)

    z = eps.mul(sigma).add_(mu)
    return z

class InteractivePainter(nn.Module) :

    def __init__(self, config):
        super(InteractivePainter, self).__init__()
        self.context_encoder = context_encoder.ContextEncoder(config)
        self.vae_encoder = vae_encoder.VAEEncoder(config)
        self.vae_decoder = vae_decoder.VAEDecoder(config)

        self.use_discriminator = config["train"]["discriminator"]["enabled"]
        if self.use_discriminator:
            self.D = discriminator.Discriminator(config)

    def forward(self, data):
        context, visual_features = self.context_encoder(data)
        mu, log_sigma = self.vae_encoder(data, context)

        z = reparameterize(mu, log_sigma)
        predictions = self.vae_decoder(z=z,
                                       context=context,
                                       visual_features=visual_features,
                                       L=data['strokes_seq'].size(1))

        return predictions, mu, log_sigma

    def forward_discriminator(self, data):
        context, visual_features = self.context_encoder(data)
        mu, log_sigma = self.vae_encoder(data, context)

        preds_with_z = self.vae_decoder(z=reparameterize(mu, log_sigma),
                                       context=context,
                                       visual_features=visual_features,
                                       L=data['strokes_seq'].size(1))


        # Sample
        preds_no_z = self.vae_decoder(z=torch.randn(data['strokes_seq'].size(0), self.d_model).to(self.device),
                                       context=context,
                                       visual_features=visual_features,
                                       L=data['strokes_seq'].size(1))


        return preds_with_z, preds_no_z