import argparse
import logging
import random

import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader

from model.networks.efdm import load_pretrained_efdm
from model.utils.parse_config import ConfigParser
from model.dataset import StrokesDataset, StylizedStrokesDataset
from model.dataloader import DataLoaderWrapper, collate_strokes
from model.training.trainer import Trainer
from model import build_model


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
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

    # Initialize wandb
    wandb.init(project=config["train"]["logging"]["project_name"], config=config)
    wandb.run.name = config["train"]["logging"]["exp_name"]
    # EFDM stylization model
    if config["stylization"]["apply"]:
        vgg_path = config["stylization"]["vgg_weights"]
        decoder_path = config["stylization"]["decoder_weights"]
        efdm = load_pretrained_efdm(vgg_path, decoder_path).eval().to(torch.device('cuda:0'))
    else:
        efdm = None
    # Collate function
    collate_fn = collate_strokes if config["stylization"]["apply"] else None
    # Train
    if config["stylization"]["apply"]:
        dataset = StylizedStrokesDataset(config, isTrain=True)
    else:
        dataset = StrokesDataset(config, isTrain=True)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=config["train"]["batch_size"],
                              shuffle=True,
                              num_workers=config["train"]["num_workers"],
                              pin_memory=False,
                              collate_fn=collate_fn)
    train_loader = DataLoaderWrapper(train_loader, efdm_model=efdm)

    # Test
    if config["stylization"]["apply"]:
        dataset_test = StylizedStrokesDataset(config, isTrain=False)
    else:
        dataset_test = StrokesDataset(config, isTrain=False)
    test_loader = DataLoader(dataset=dataset_test,
                             batch_size=config["train"]["test_batch_size"],
                             num_workers=config["train"]["num_workers"],
                             shuffle=True,
                             pin_memory=False,
                             collate_fn=collate_fn)
    test_loader = DataLoaderWrapper(test_loader, efdm_model=efdm)

    logging.info(f'Dataset stats: Train {len(dataset)} samples, Test : {len(dataset_test)} samples')

    # Create model
    model = build_model(config=config)
    # model = nn.DataParallel(model)
    model.cuda()

    params = count_parameters(model)
    logging.info(f'Number of trainable parameters: {params / 10 ** 6}M')

    # Create
    start_epoch = 1
    trainer = Trainer(config, model, train_loader, test_loader)
    if config["train"]["auto_resume"]["active"]:
        start_epoch = model.load_checkpoint(config["train"]["auto_resume"]["resume_path"]) + 1
    max_epochs = config["train"]["n_epochs"]
    wandb.watch(model)
    for ep in range(start_epoch, max_epochs + 1):
        # Print
        logging.info('=' * 50)
        logging.info(f'Epoch: {ep} / {config["train"]["n_epochs"]}')

        train_stats = trainer.train_one_epoch(model, ep)
        wandb.log(train_stats)

        # Eval
        if ep % config["train"]["logging"]["eval_every"] == 0 or (ep == max_epochs):
            logging.info('=' * 50)
            logging.info('** EVALUATION **')
            torch.cuda.empty_cache()
            test_logs = trainer.evaluate(model)
            wandb.log(test_logs)

        # Save ckpt
        if (ep % config["train"]["logging"]["save_freq"] == 0):
            model.save_checkpoint(epoch=ep)

    # Save final model
    model.save_checkpoint(epoch=ep, filename='latest')
