import yaml
import os
import glob
from pathlib import Path
import torch


class ConfigParser:
    def __init__(self, config_path, isTrain=True):
        self.isTrain = isTrain
        self.config_path = config_path

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def create_exp_name(self):
        w_pos = self.config["train"]["losses"]["reconstruction"]["weight"]["position"]
        w_size = self.config["train"]["losses"]["reconstruction"]["weight"]["size"]
        w_theta = self.config["train"]["losses"]["reconstruction"]["weight"]["theta"]
        w_color = self.config["train"]["losses"]["reconstruction"]["weight"]["color"]
        w_ref_color = self.config["train"]["losses"]["reference_img"]["color"]["weight"]
        # w_pos_color = self.config["train"]["losses"]["reference_img"]["pos_color"]["weight"]
        w_kl = self.config["train"]["losses"]["kl"]["weight"]

        name = f'VAE-pos{str(w_pos)}-gt_col{str(w_color)}-ref_col{str(w_ref_color)}-kl{str(w_kl)}'

        self.config["train"]["logging"]["exp_name"] += name

    def parse_config(self):
        if self.isTrain:
            assert self.config["model"]["ctx_z"] == 'proj' or self.config["model"]["ctx_z"] == 'cat'
            assert self.config["dataset"]["partition"] == 'ade_dataset' or self.config["dataset"][
                "partition"] == 'oxford_pet_dataset'

            self.create_exp_name()
            self.config["train"]["logging"]["checkpoint_path"] = os.path.join(
                self.config["train"]["logging"]["checkpoint_path"], self.config["train"]["logging"]["exp_name"])
            f = os.path.join(self.config["train"]["logging"]["checkpoint_path"])
            if os.path.exists(f):
                print(f'Auto Resume from : {f}')
                files = sorted(glob.glob(os.path.join(f, '*.pth.tar')))
                if len(files) > 0:
                    self.config["train"]["auto_resume"]["active"] = True
                    self.config["train"]["auto_resume"]["resume_path"] = os.path.join(f, files[-1])

            # self.config["train"]["logging"]["log_render_path"] = os.path.join(self.config["train"]["logging"]["checkpoint_path"], 'renders')
            id_device = self.config["train"]["gpu_id"]
        else:
            id_device = self.config["gpu_id"]

        if id_device >= 0:
            self.config["device"] = torch.device(f'cuda:{id_device}')
        else:
            self.config["device"] = torch.device('cpu')

    def crate_directory_output(self):
        Path(self.config["train"]["logging"]["checkpoint_path"]).mkdir(parents=True, exist_ok=True)
        # Path(self.config["train"]["logging"]["log_render_path"]).mkdir(parents=True, exist_ok=True)

    def get_config(self):
        return self.config
