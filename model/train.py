from utils.parse_config import ConfigParser
import argparse
from model import InteractivePainter
from dataset import StrokesDataset
from training.training import Trainer
from torch.utils.data import DataLoader
from torch import device
import torch
import pickle
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()

    # Create config
    c_parser = ConfigParser(args)
    c_parser.parse_config(args)
    c_parser.crate_directory_output()
    config = c_parser.get_config()


    # Device
    device = device(f'cuda:{config["train"]["gpu_id"]}')

    # Create dataset
    dataset = StrokesDataset(config)
    dataloader = DataLoader(dataset=dataset, batch_size=config["train"]["batch_size"], shuffle=True)

    print(f'Dataset length: {len(dataset)}')
    print(f'Dataloader: {len(dataloader)}')

    # Create model
    model = InteractivePainter(config)
    model.to(device)

    def count_parameters(net):
        return sum(p.numel() for p in net.parameters() if p.requires_grad)

    params = count_parameters(model)
    print(f'Number of trainable parameters: {params / 10**6}')

    # Create
    trainer = Trainer(config, model, dataloader, device=device)

    tot = {'mse_loss' : [],
           'kl' : [],
           'loss' : []}
    for ep in range(config["train"]["n_epochs"]):
        stats = trainer.train_one_epoch(model, ep)
        if ep % config["train"]["save_freq"] == 0:
            trainer.save_checkpoint(model, filename=f'epoch_{ep}')

        print('='*50)
        print(f'Epoch: {ep} / {config["train"]["n_epochs"]}')
        for k, vals in stats.items():
            mean = torch.mean(torch.tensor(vals))
            tot[k].append(mean.item())
            print(f'{k} = {mean.item()}')

    with open(os.path.join(config["train"]["checkpoint_path"],'logs.pkl'), 'wb') as f:
        pickle.dump(tot, f)
    trainer.save_checkpoint(model)




