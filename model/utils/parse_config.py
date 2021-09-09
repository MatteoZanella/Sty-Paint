import yaml
import os
from pathlib import Path

class ConfigParser:
    def __init__(self, args, is_train=True):
        self.is_train = is_train
        self.config_path = args.config

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def parse_config(self, args):

        # Add total sequence length
        if self.is_train:
            self.config["train"]["checkpoint_path"] = os.path.join(self.config["train"]["checkpoint_path"], args.exp_name)
            if args.cat_x_z:
                self.config["model"]["ctx_z"] = 'cat'
            if args.sigm:
                self.config["model"]["activation_last_layer"] = 'sigmoid'
            #assert self.config["model"]["ctx_z"] == 'proj' or self.config["model"]["ctx_z"] == 'cat'
            #assert self.config["model"]["activation_last_layer"] == 'identity' or self.config["model"]["activation_last_layer"] == 'sigmoid'
            self.config["dataset"]["debug"] = args.debug
        else:
            self.config["dataset"]["debug"] = False

    def crate_directory_output(self):

        Path(self.config["train"]["checkpoint_path"]).mkdir(parents=True, exist_ok=True)


    def get_config(self):
        return self.config

