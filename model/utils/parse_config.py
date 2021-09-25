import yaml
import os
from pathlib import Path
import torch
import glob

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
            self.config["train"]["logging"]["checkpoint_path"] = os.path.join(self.config["train"]["logging"]["checkpoint_path"], args.exp_name)
            f = os.path.join(self.config["train"]["logging"]["checkpoint_path"], 'latest.pth.tar')
            if os.path.exists(f):
                print(f'Auto Resume from : {f}')
                self.config["train"]["auto_resume"]["active"] = True
                self.config["train"]["auto_resume"]["resume_path"] = f

            self.config["train"]["logging"]["log_render_path"] = os.path.join(self.config["train"]["logging"]["checkpoint_path"], 'renders')

        if self.config["train"]["gpu_id"] >= 0:
            self.config["device"] = torch.device(f'cuda:{self.config["train"]["gpu_id"]}')
        else:
            self.config["device"] = torch.device('cpu')

    def crate_directory_output(self):
        Path(self.config["train"]["logging"]["checkpoint_path"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["train"]["logging"]["log_render_path"]).mkdir(parents=True, exist_ok=True)

    def get_config(self):
        return self.config

