import os
import argparse
import torch
import torch.optim as optim
import numpy as np

from decomposition import utils
from decomposition.painter import Painter

import pickle
from datetime import datetime


def get_args():
    # settings
    parser = argparse.ArgumentParser(description='STROKES DECOMPOSITION')
    parser.add_argument('--output_path', required=True, type=str, help='output')
    parser.add_argument('--painter_config', default='./decomposition/painter_config.yaml')
    parser.add_argument('--image_path', default='/home/eperuzzo/ade20k_outdoors/images/training/', type=str, help='Image path')
    parser.add_argument('--plot_loss', default=True)
    parser.add_argument('--gpu_id', default=0, type=int, help='GPU index')
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu_id))
    else:
        device = torch.device('cpu')

    painter_config = utils.load_painter_config(args.painter_config)
    source_images = [
        'ADE_train_00000583.jpg',
        'ADE_train_00000568.jpg',
        'ADE_train_00000555.jpg',
        'ADE_train_00000590.jpg',
        'ADE_train_00001487.jpg']

    k = 1
    for source_image in source_images:
        start = datetime.now()
        print('Processing image: {}, {}/{}'.format(source_image, k, len(source_images)))
        k += 1
        # TODO
        name = source_image.split('.')[0]
        tmp_output_path = os.path.join(args.output_path, name)
        os.makedirs(tmp_output_path, exist_ok=True)

        # --------------------------------------------------------------------------------------------------------------
        # Decomposition
        painter_config.img_path = os.path.join(args.image_path, source_image)
        pt = Painter(args=painter_config)
        strokes = pt.train()
        pt._save_stroke_params(strokes, path=tmp_output_path)
        pt.inference(strokes, output_path=os.path.join(tmp_output_path, 'original'), save_video=True)

        # --------------------------------------------------------------------------------------------------------------
        # Save loss curves dictonary and figures
        elapsed = datetime.now() - start
        logs = pt.loss_dict
        logs['elapsed_time'] = elapsed
        with open(os.path.join(tmp_output_path, 'logs.pkl'), 'wb') as f:
            pickle.dump(logs, f, pickle.HIGHEST_PROTOCOL)
        if args.plot_loss:
            utils.plot_loss_curves(logs, tmp_output_path)

