import torch
from torch.optim import AdamW
import torch.nn as nn
from dataset import ToDevice
from training.losses import KLDivergence
import os

class Trainer:

    def __init__(self, config, model, dataloader, device):

        self.checkpoint_path = config["train"]["checkpoint_path"]
        self.optimizer = AdamW(params=model.parameters(), lr=config["train"]["lr"], weight_decay=config["train"]['wd'])
        self.dataloader = dataloader
        self.MSELoss = nn.MSELoss()
        self.KLDivergence = KLDivergence()
        self.move_to_device = ToDevice(device)

    def save_checkpoint(self, model, filename=None):

        if filename is None:
            path = os.path.join(self.checkpoint_path, "latest.pth.tar")
        else:
            path = os.path.join(self.checkpoint_path, f"{filename}_.pth.tar")

        torch.save({"model": model.state_dict(),
                    "optimizer": self.optimizer.state_dict()}, path)

        print(f'Model saved at {path}')

    def load_checkpoint(self, model, filename=None):
        #TODO
        pass

    def train_one_epoch(self, model, epoch):

        # Set trainign mode
        model.train()

        logs = {
            "mse_loss" : [],
            "kl" : [],
            "loss" : []
        }
        for idx, batch in enumerate(self.dataloader):
            batch = self.move_to_device.move_dict_to(batch)
            labels = batch['sequence']['strokes']

            predictions, mu, log_sigma = model(batch)
            mse_loss = self.MSELoss(predictions, labels)
            kl_div = self.KLDivergence(mu, log_sigma)

            import pdb
            pdb.set_trace()

            loss = mse_loss + kl_div

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            logs["mse_loss"].append(mse_loss.item())
            logs["kl"].append(kl_div.item())
            logs["loss"].append(loss)


        return logs