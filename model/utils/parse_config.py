import yaml
import os
from pathlib import Path
import torch

class ConfigParser:
    def __init__(self, config_path, isTrain=True):
        self.isTrain = isTrain
        self.config_path = config_path

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def create_exp_name(self):
        m_type = self.config["model"]["model_type"]
        w_pos = self.config["train"]["losses"]["reconstruction"]["weight"]["position"]
        w_size = self.config["train"]["losses"]["reconstruction"]["weight"]["size"]
        w_theta = self.config["train"]["losses"]["reconstruction"]["weight"]["theta"]
        w_color = self.config["train"]["losses"]["reconstruction"]["weight"]["color"]
        w_ref_color =self.config["train"]["losses"]["reference_img"]["color"]["weight"]
        w_ref_render =  self.config["train"]["losses"]["reference_img"]["render"]["weight"]
        #w_pos_color = self.config["train"]["losses"]["reference_img"]["pos_color"]["weight"]
        w_kl = self.config["train"]["losses"]["kl"]["weight"]

        name = f'{m_type}-pos{str(w_pos)}-gt_col{str(w_color)}-ref_col{str(w_ref_color)}-kl{str(w_kl)}'

        self.config["train"]["logging"]["exp_name"] += name


    def parse_config(self):
        if self.isTrain:
            assert self.config["model"]["context_encoder"]["canvas_strokes"] == 'proj' or self.config["model"]["context_encoder"]["canvas_strokes"] == 'add'
            assert self.config["model"]["ctx_z"] == 'proj' or self.config["model"]["ctx_z"] == 'cat'
            assert self.config["dataset"]["partition"] == 'ade_dataset' or self.config["dataset"]["partition"] == 'oxford_pet_dataset'
            # assert self.config["model"]["encoder_pe"] == 'sine' or self.config["model"]["encoder_pe"] == '3dsine'
            # assert self.config["model"]["decoder_pe"] == 'sine' or self.config["model"]["decoder_pe"] == 'learnable'

            self.create_exp_name()
            self.config["train"]["logging"]["checkpoint_path"] = os.path.join(self.config["train"]["logging"]["checkpoint_path"], self.config["train"]["logging"]["exp_name"])
            f = os.path.join(self.config["train"]["logging"]["checkpoint_path"], 'latest.pth.tar')
            if os.path.exists(f):
                print(f'Auto Resume from : {f}')
                self.config["train"]["auto_resume"]["active"] = True
                self.config["train"]["auto_resume"]["resume_path"] = f

            #self.config["train"]["logging"]["log_render_path"] = os.path.join(self.config["train"]["logging"]["checkpoint_path"], 'renders')
            id_device = self.config["train"]["gpu_id"]
        else:
            id_device = self.config["gpu_id"]

        if id_device >= 0:
            self.config["device"] = torch.device(f'cuda:{id_device}')
        else:
            self.config["device"] = torch.device('cpu')


    def crate_directory_output(self):
        Path(self.config["train"]["logging"]["checkpoint_path"]).mkdir(parents=True, exist_ok=True)
        #Path(self.config["train"]["logging"]["log_render_path"]).mkdir(parents=True, exist_ok=True)

    def get_config(self):
        return self.config