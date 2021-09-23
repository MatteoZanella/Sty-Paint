import argparse
import os
import logging

from model.utils.parse_config import ConfigParser
from model.model import InteractivePainter
from model.only_vae import OnlyVAE
from model.dataset import StrokesDataset
from model.training.trainer import Trainer
from torch.utils.data import DataLoader
import wandb

def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


if __name__ == '__main__':
    os.environ["WANDB_API_KEY"] = "5745461314f5f4abb2c957bb991d2df97144ba06"

    # Extra parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type= str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ctx_z", type=str, choices=['cat', 'proj'])
    parser.add_argument("--only_vae", action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Create config
    c_parser = ConfigParser(args)
    c_parser.parse_config(args)
    config = c_parser.get_config()
    c_parser.crate_directory_output()
    print(config)

    # Initialize wandb
    wandb.init(project='Brushstrokes-Generation', config=config)
    wandb.run.name = args.exp_name

    # Create dataset_acquisition
    device = config["device"]
    # Train
    dataset = StrokesDataset(config, isTrain=True)
    train_loader = DataLoader(dataset=dataset, batch_size=config["train"]["batch_size"], shuffle=True, num_workers=config["train"]["num_workers"])
    # Test
    dataset_test = StrokesDataset(config, isTrain=False)
    test_loader = DataLoader(dataset=dataset_test, batch_size=1, shuffle=True)

    logging.info(f'Dataset stats: Train {len(dataset)} samples, Test : {len(dataset_test)} samples')

    # Create model
    if args.only_vae:
        logging.info('VAE ONLY, NO CONTEXT')
        model = OnlyVAE(config)
        model.to(device)
    else:
        logging.info('FULL MODEL')
        model = InteractivePainter(config)
        model.to(device)

    params = count_parameters(model)
    logging.info(f'Number of trainable parameters: {params / 10**6}M')

    # Create
    trainer = Trainer(config, model, train_loader, test_loader)
    max_epochs = config["train"]["n_epochs"]

    #generator = GenerateStorkes(config["render"]["painter_config"], output_path=config["train"]["logging"]["log_render_path"])

    wandb.watch(model)
    for ep in range(1, max_epochs+1):
        # Print
        logging.info('=' * 50)
        logging.info(f'Epoch: {ep} / {config["train"]["n_epochs"]}')

        train_stats = trainer.train_one_epoch(model, ep)
        test_logs = trainer.evaluate(model, ep)

        # Log
        wandb.log(train_stats)
        wandb.log(test_logs)

        # Save ckpt
        if (ep % config["train"]["logging"]["save_freq"] == 0) :
            trainer.save_checkpoint(model, epoch=ep)

    # Save final model
    trainer.save_checkpoint(model, epoch=ep, filename='latest')







