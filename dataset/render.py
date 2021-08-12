import argparse
import os
import torch

from decomposition.painter import Painter
import torch
import sorting.utils as sutils
import matplotlib.pyplot as plt
import numpy as np

def get_args():
    # settings
    parser = argparse.ArgumentParser(description='STYLIZED NEURAL PAINTING')
    parser.add_argument('--data_path', default='./images/training/')
    parser.add_argument('--img_path', default='/home/eperuzzo/ade20k_outdoors/images/training/ADE_train_00001487.jpg')
    parser.add_argument('--renderer', default='oilpaintbrush')
    parser.add_argument('--canvas_color', default='black')
    parser.add_argument('--canvas_size', default=512)
    parser.add_argument('--keep_aspect_ratio', default=False, type=bool)
    parser.add_argument('--max_m_strokes', default=500, type=int)
    parser.add_argument('--max_divide', default=5, type=int)
    parser.add_argument('--beta_L1', default=1.0, type=float)
    parser.add_argument('--with_ot_loss', default=False, type=bool)
    parser.add_argument('--beta_ot', default=0.1, type=float)
    parser.add_argument('--net_G', default='zou-fusion-net', type=str)
    parser.add_argument('--renderer_checkpoint_dir', default='./checkpoints_G_oilpaintbrush', type=str)
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--output_dir', default='./imagenet_output', type=str)
    parser.add_argument('--disable_preview', default=True, type=bool)
    parser.add_argument('--clamp_w_h', default=0.9, type=float)
    parser.add_argument('--gpu_id', default=0, type=int)
    # parser.add_argument('--alpha_spacing', default=0.2)
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    # Decide which device we want to run on
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu_id))
    else:
        device = torch.device('cpu')

    source_images = [
        'ADE_train_00000583.jpg',
        'ADE_train_00000568.jpg',
        'ADE_train_00000555.jpg',
        'ADE_train_00000590.jpg',
        'ADE_train_00001487.jpg']  # [f for f in os.listdir(args.data_path) if f.endswith(('.jpg', '.png'))]
    k = 1
    pt = Painter(args=args)

    path_strokes = '/home/eperuzzo/brushstrokes/strokes_params/ADE_train_00001487/'
    strokes_loader = sutils.StrokesLoader(path_strokes)
    strokes, _ = strokes_loader.load_strokes()

    path_idx = '/home/eperuzzo/brushstrokes/prova2/ADE_train_00001487/lkh/index/lkh_col1_area1_pos0_cl5_sal0.pkl'
    idx = sutils.load_pickle(path_idx)
    idx = np.array(idx)
    print(f'storkes {strokes.shape}')
    print(f'IDx: {idx.shape}')

    v = strokes[:, idx, :]
    print(f'v : {v.shape}')

    origial, _ = pt._render(strokes, '', False, False)


