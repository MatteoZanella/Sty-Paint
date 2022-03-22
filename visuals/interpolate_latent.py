import argparse
import os
import matplotlib.pyplot as plt
import torch
import pandas as pd

import sys
sys.path.insert(1, '../')

from model.utils.utils import dict_to_device
from model.utils.parse_config import ConfigParser
from model import build_model
from model.dataset import EvalDataset

from dataset_acquisition.decomposition.painter import Painter
from dataset_acquisition.decomposition.utils import load_painter_config

from evaluation.tools import produce_visuals


if __name__ == '__main__' :
    # Extra parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config",
                        default='/home/eperuzzo/brushstrokes-generation/configs/eval/eval_oxford.yaml')
    parser.add_argument("--files_setup", default='/home/eperuzzo/config.csv')
    parser.add_argument("--output_path", type=str, default='./results/')
    args = parser.parse_args()

    # Create config
    c_parser = ConfigParser(args.config, isTrain=False)
    c_parser.parse_config()
    config = c_parser.get_config()

    # Test
    dataset_test = EvalDataset(config, isTrain=False)
    print(f'Test : {len(dataset_test)} samples')

    # ======= Create Models ========================
    # Renderer (Stylized Neural Painting)
    render_config = load_painter_config(config["renderer"]["painter_config"])
    renderer = Painter(args=render_config)

    # load checkpoint, update model config based on the stored config
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    config.update(dict(model=ckpt["config"]["model"]))

    model = build_model(config)
    msg = model.load_state_dict(ckpt["model"], strict=False)
    print(f'==> Loading model form {args.checkpoint}, with : {msg}')
    model.cuda()
    model.eval()

    files = pd.read_csv(args.files_setup)
    for idx, row in files.iterrows() :
        filename = row["filename"]
        ts = row["ts"]
        print(f'==> Processing img : {filename}')
        os.makedirs(args.output_path, exist_ok=True)

        batch, _, _ = dataset_test.sample(filename, ts)

        # batch =
        data = dict_to_device(batch, to_skip=['strokes', 'time_steps'])
        targets = data['strokes_seq']
        starting_point = batch['canvas'][0].permute(1, 2, 0).cpu().numpy()
        img = (batch['img'][0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')

        num_samples = 10
        w_ii = torch.linspace(0, 1, num_samples)

        num_z = 5
        Z_start = torch.randn(1, 256)

        Z = torch.empty(num_z, num_samples, 256)
        for j in range(num_z):
            Z_j = torch.randn(1, 256)
            for ii in range(num_samples):
                Z[j, ii] = Z_start * (1 - w_ii[ii]) + Z_j * w_ii[ii]
        # Our model
        for j in range(num_z):
            with torch.no_grad():
                fake_data_random = model.sample(data, Z[j].cuda())

            for ii in range(fake_data_random.shape[0]):
                params = fake_data_random[ii].unsqueeze(0).cpu()

                visual = produce_visuals(params,
                                        renderer=renderer,
                                        starting_canvas=starting_point,
                                        ctx=batch['strokes_ctx'])

                plt.imsave(os.path.join(args.output_path,
                                        f'{filename}_n_{str(j).zfill(2)}_s_{str(ii).zfill(3)}.png'),
                           visual)