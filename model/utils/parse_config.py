import yaml
import os
from pathlib import Path
import torch

class ConfigParser:
    def __init__(self, args, is_train=True):
        self.is_train = is_train
        self.config_path = args.config

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def parse_config(self, args):

        assert self.config["model"]["ctx_z"] == 'proj' or self.config["model"]["ctx_z"] == 'cat'

        if args.only_vae:
            self.config["model"]["only_vae"] = True
            self.config["model"]["ctx_z"] = None

        if self.is_train:
            self.config["train"]["checkpoint_path"] = os.path.join(self.config["train"]["checkpoint_path"], args.exp_name)
            self.config["train"]["train_render"] = os.path.join(self.config["train"]["checkpoint_path"], 'renders')

        self.config["device"] = torch.device(f'cuda:{self.config["train"]["gpu_id"]}')

    def crate_directory_output(self):
        Path(self.config["train"]["checkpoint_path"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["train"]["train_render"]).mkdir(parents=True, exist_ok=True)

    def get_config(self):
        return self.config

