import argparse
import os

from model.utils.parse_config import ConfigParser
from model.model import InteractivePainter
from model.only_vae import OnlyVAE
from model.dataset import StrokesDataset
from model.training.training import Trainer
from torch.utils.data import DataLoader
from model.generate import GenerateStorkes
import wandb

def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


if __name__ == '__main__':
    os.environ["WANDB_API_KEY"] = "5745461314f5f4abb2c957bb991d2df97144ba06"

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type= str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ctx_z", type=str, choices=['cat', 'proj'])
    parser.add_argument("--only_vae", action='store_true')
    args = parser.parse_args()

    # Create config
    c_parser = ConfigParser(args)
    c_parser.parse_config(args)
    c_parser.crate_directory_output()
    config = c_parser.get_config()
    print(config)

    # Initialize wandb
    wandb.init(project='Brushstrokes-Generation', config=config)
    wandb.run.name = args.exp_name

    # Create dataset
    device = config["device"]
    # Train
    dataset = StrokesDataset(config, isTrain=True)
    train_loader = DataLoader(dataset=dataset, batch_size=config["train"]["batch_size"], shuffle=True)
    # Test
    dataset_test = StrokesDataset(config, isTrain=False)
    test_loader = DataLoader(dataset=dataset_test, batch_size=config["train"]["batch_size"], shuffle=True)

    print(f'Dataset stats: Train {len(dataset)} samples, Test : {len(dataset_test)} samples')

    # Create model
    if args.only_vae:
        print('VAE ONLY, NO CONTEXT')
        model = OnlyVAE(config)
        model.to(device)
    else:
        print('FULL MODEL')
        model = InteractivePainter(config)
        model.to(device)

    params = count_parameters(model)
    print(f'Number of trainable parameters: {params / 10**6}M')

    # Create
    trainer = Trainer(config, model, train_loader)
    max_epochs = config["train"]["n_epochs"]

    generator = GenerateStorkes(config["render"]["painter_config"], output_path=config["train"]["train_render"])

    for ep in range(1, max_epochs+1):
        # Print
        print('=' * 50)
        print(f'Epoch: {ep} / {config["train"]["n_epochs"]}')
        train_stats = trainer.train_one_epoch(model)

        if (ep % config["train"]["save_freq"] == 0):
            trainer.save_checkpoint(model, filename=f'epoch_{ep}')
        if ep % config["render"]["freq"] == 0:
            render, _ = generator.generate_and_render(model, test_loader)
            generator.save_renders(render, filename=f'epoch_{ep}')
            # Wandb logging
            wandb.log({"render" : wandb.Image(render, caption=f"render_ep_{ep}")})

        # Log
        wandb.log(train_stats)

    # Save final model
    trainer.save_checkpoint(model, filename='latest')







