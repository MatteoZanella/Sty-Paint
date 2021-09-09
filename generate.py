import argparse

from model.utils.parse_config import ConfigParser
from model.model import InteractivePainter
from dataset.decomposition.painter import Painter
from dataset.decomposition.utils import load_painter_config
from model.dataset import StrokesDataset
from torch.utils.data import DataLoader
import torch
from torch import device
from model.utils.utils import dict_to_device
from pathlib import Path
import os
import numpy as np

# Debug
import matplotlib.pyplot as plt

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--len", type=int, default=8)
    parser.add_argument("--config", default='./configs/train/config.yaml')
    parser.add_argument("--output_path", default='/home/eperuzzo/brushstrokes/tmp_out/')
    parser.add_argument("--ckpt", default='/home/eperuzzo/brushstrokes/ckpt/t1/epoch_499_.pth.tar')
    parser.add_argument('--painter_config', default='./configs/decomposition/painter_config_local.yaml')
    args = parser.parse_args()

    # Create config
    c_parser = ConfigParser(args, is_train=False)
    c_parser.parse_config(args)
    config = c_parser.get_config()


    # Device
    device = device(f'cuda:{config["train"]["gpu_id"]}')

    # Create dataset
    dataset = StrokesDataset(config, split='test')
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    print(f'Dataset lenght : {len(dataset)}')
    # Create model
    model = InteractivePainter(config)
    model.to(device)

    print("Restore weights..")
    state_dict = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state_dict["model"])

    # Define Painter
    painter_config = load_painter_config(args.painter_config)
    pt = Painter(args=painter_config)

    # Generate
    model.eval()
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_path, 'original')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_path, 'generated')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_path, 'context')).mkdir(parents=True, exist_ok=True)

    for i, data in enumerate(dataloader):
        t_C, t, t_T = data['time_steps']
        strokes = data['strokes']

        context = strokes[:, t_C:t, :]
        seq = strokes[:, t: t_T, :]

        data = dict_to_device(data, device, to_skip=['time_steps', 'strokes'])
        pred_strokes = model.generate(data, args.len)

        sp, _ = pt.inference(strokes[:, :t, :])
        plt.imsave(os.path.join(args.output_path, 'starting_point.png'), sp)

        pt.inference(strokes=context, output_path=os.path.join(args.output_path, 'context'), save_jpgs=True, save_video=True)
        pt.inference(strokes=seq, output_path=os.path.join(args.output_path, 'original'), save_jpgs=True, save_video=True)

        gen = pred_strokes.detach().cpu().numpy()
        pt.inference(strokes=gen, output_path=os.path.join(args.output_path, 'generated'),
                     save_jpgs=True, save_video=True)











