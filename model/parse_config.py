import yaml
from pathlib import Path
import os
from types import SimpleNamespace

class ConfigParser:
    def __init__(self, path : str):

        with open(path, 'r') as f:
            self.config = yaml.safe_load(f)

    def parse_config(self):

        self.config['model']['d_model'] = 512 + self.config['model']['n_strokes_params']

    def get_config(self):
        return self.config

    def crate_directory_output(self):

        Path(self.config["logging"]["output_directory"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["logging"]["save_root_directory"]).mkdir(parents=True, exist_ok=True)

