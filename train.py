import argparse
import os
import logging
import numpy as np
import torch

from model.utils.parse_config import ConfigParser
from model.dataset import StrokesDataset
from model.training.trainer import Trainer
from torch.utils.data import DataLoader
from model import build_model


import wandb

def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Extra parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Create config
    c_parser = ConfigParser(config_path=args.config)
    c_parser.parse_config()
    config = c_parser.get_config()
    c_parser.crate_directory_output()
    print(config)

    # Seed
    seed = config["train"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    # Initialize wandb
    os.environ["WANDB_API_KEY"] = config["train"]["logging"]["wandb_api_key"]
    wandb.init(project=config["train"]["logging"]["project_name"], config=config)
    wandb.run.name = config["train"]["logging"]["exp_name"]

    # Train
    dataset = StrokesDataset(config, isTrain=True)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=config["train"]["batch_size"],
                              shuffle=True,
                              num_workers=config["train"]["num_workers"],
                              pin_memory=False)

    # Test
    dataset_test = StrokesDataset(config, isTrain=False)
    test_loader = DataLoader(dataset=dataset_test,
                             batch_size=64,
                             num_workers=config["train"]["num_workers"],
                             shuffle=True,
                             pin_memory=False)

    logging.info(f'Dataset stats: Train {len(dataset)} samples, Test : {len(dataset_test)} samples')

    # Create model
    model = build_model(config=config)
    #model = nn.DataParallel(model)
    model.cuda()

    params = count_parameters(model)
    logging.info(f'Number of trainable parameters: {params / 10**6}M')

    # Create
    start_epoch = 1
    trainer = Trainer(config, model, train_loader, test_loader)
    model.train_setup(n_iters_per_epoch=len(train_loader))
    if config["train"]["auto_resume"]["active"]:
        start_epoch = model.load_checkpoint(config["train"]["auto_resume"]["resume_path"])+1
    max_epochs = config["train"]["n_epochs"]
    wandb.watch(model)
    for ep in range(start_epoch, max_epochs+1):
        # Print
        logging.info('=' * 50)
        logging.info(f'Epoch: {ep} / {config["train"]["n_epochs"]}')

        train_stats = trainer.train_one_epoch(model, ep)
        wandb.log(train_stats)

        # Eval
        if ep % config["train"]["logging"]["eval_every"] == 0 or (ep == max_epochs):
            logging.info('=' * 50)
            logging.info('** EVALUATION **')
            test_logs = trainer.evaluate(model)
            wandb.log(test_logs)

        # Save ckpt
        if (ep % config["train"]["logging"]["save_freq"] == 0) :
            model.save_checkpoint(epoch=ep)

    # Save final model
    model.save_checkpoint(epoch=ep, filename='latest')