from .modeltype import VAE

def build_model(config) :
    return VAE.VAEModel(config)