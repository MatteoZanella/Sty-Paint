import yaml
from pathlib import Path

class ConfigParser:
    def __init__(self, args):
        self.exp_name = args.exp_name
        self.config_path = args.config

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def parse_config(self):

        print(f'Model features  : {self.config["model"]["d_model"]}')
        # Add total sequence length
        self.config['dataset']['total_length'] = self.config["dataset"]["sequence_length"] + self.config["dataset"]["context_length"]

    def crate_directory_output(self):

        Path(self.config["train"]["checkpoint_path"]).mkdir(parents=True, exist_ok=True)


    def get_config(self):
        return self.config

