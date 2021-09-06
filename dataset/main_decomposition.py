import os
import argparse
import torch

from decomposition import utils
from decomposition.painter import Painter

import pickle
from datetime import datetime
import pandas as pd
import numpy as np


def get_args():
    # settings
    parser = argparse.ArgumentParser(description='STROKES DECOMPOSITION')
    parser.add_argument('--output_path', required=True, type=str, help='output')
    parser.add_argument('--csv_file', required=True, type=str, help='Image path')
    parser.add_argument('--painter_config', default='./decomposition/painter_config.yaml')
    parser.add_argument('--plot_loss', default=False)
    parser.add_argument('--gpu_id', default=0, type=int, help='GPU index')
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu_id))
    else:
        device = torch.device('cpu')

    # Define Painter
    painter_config = utils.load_painter_config(args.painter_config)
    pt = Painter(args=painter_config)

    df = pd.read_csv(args.csv_file)
    print(f'Total Number of images to process in this chunk: {len(df)}')

    errors = []
    for index, row in df.iterrows():
        try:
            start = datetime.now()
            print('Processing image: {}, {}/{}'.format(row['Images'], index, len(df)))
            img_path = row['Images']

            name = os.path.basename(img_path).split('.')[0]
            tmp_output_path = os.path.join(args.output_path, name)
            os.makedirs(tmp_output_path, exist_ok=True)
            # --------------------------------------------------------------------------------------------------------------
            # Decomposition
            painter_config.img_path = img_path
            strokes = pt.train()
            pt._save_stroke_params(strokes, path=tmp_output_path)
            final_img, alphas = pt.inference(strokes)
            np.savez_compressed(os.path.join(tmp_output_path, 'alpha.npz'), alpha=alphas)
            # --------------------------------------------------------------------------------------------------------------
            # Save loss curves dictonary and figures
            elapsed = datetime.now() - start
            logs = pt.loss_dict
            logs['elapsed_time'] = elapsed
            with open(os.path.join(tmp_output_path, 'logs.pkl'), 'wb') as f:
                pickle.dump(logs, f, pickle.HIGHEST_PROTOCOL)
            if args.plot_loss:
                utils.plot_loss_curves(logs, tmp_output_path)
        except:
            img_name = row['Images']
            print(f'Error occured processing {img_name}')
            errors.append(img_name)

