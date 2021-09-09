from utils.parse_config import ConfigParser
import argparse
from model import InteractivePainter
from dataset import StrokesDataset
from training.training import Trainer
from torch.utils.data import DataLoader
from torch import device
import pickle
import os

# Debug
import matplotlib.pyplot as plt

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
    dataset = StrokesDataset(config, split='train')
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

    max_epochs = config["train"]["n_epochs"]
    tot_iter = {'mse' : [],
                'kl' : []}

    for ep in range(max_epochs):
        print('=' * 50)
        print(f'Epoch: {ep} / {config["train"]["n_epochs"]}')

        mse, kl = trainer.train_one_epoch(model)

        if ep % config["train"]["save_freq"] == 0 or ep == max_epochs-1:
            trainer.save_checkpoint(model, filename=f'epoch_{ep}')

        for val in mse:
            tot_iter['mse'].append(val)
        for val in kl:
            tot_iter['kl'].append(val)

    n_iters = range(1, len(tot_iter['mse'])+1)

    # save loss
    f = plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(n_iters, tot_iter['mse'])
    plt.title('MSE')
    plt.subplot(1, 2, 2)
    plt.plot(n_iters, tot_iter['kl'])
    plt.title('KL')
    plt.savefig(os.path.join(config["train"]["checkpoint_path"], 'losses.png'))

    with open(os.path.join(config["train"]["checkpoint_path"],'logs.pkl'), 'wb') as f:
        pickle.dump(tot_iter, f)




