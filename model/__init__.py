from .modeltype import GAN, VAE_GAN, VAE

def build_model(config) :
    if config["model"]["model_type"] == "vae":
        model = VAE.VAEModel(config)
    elif config["model"]["model_type"] == "vae-gan":
        model = VAE_GAN.VAEGANModel(config)
    elif config["model"]["model_type"] == "gan":
        model = GAN.GANModel(config)
    else:
        raise NotImplementedError(f'Model type: {config["model"]["model_type"]} is invalid. Model type can be vae, vag-gan, gan')

    return model