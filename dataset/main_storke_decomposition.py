import argparse
from graph_segmentation import Graph, GraphBuilder
import cv2
import torch

torch.cuda.current_device()
import torch.optim as optim

import numpy as np
import utils
import torch
from new_painter_renderer import NewPainter
import os

from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

def load_segmentation(path, cw):
    sm = cv2.imread(path)
    sm = cv2.resize(sm, (cw, cw), interpolation=cv2.INTER_NEAREST)
    sm = sm[:, :, 0]

    return sm


def _normalize(x, width):
    return (int)(x * (width - 1) + 0.5)


def optimize_x(pt, clamp_w_h):
    pt._load_checkpoint()
    pt.net_G.eval()
    pt._make_output_dir()
    pt.load_style_image()
    print('begin drawing...')

    clamp_schedule = {2:0.4, 3:0.3, 4:0.25, 5:0.25}
    PARAMS = np.zeros([1, 0, pt.rderr.d+2], np.float32)

    if pt.rderr.canvas_color == 'white':
        CANVAS_tmp = torch.ones([1, 3, 128, 128]).to(device)
    else:
        CANVAS_tmp = torch.zeros([1, 3, 128, 128]).to(device)

    for pt.m_grid in pt.manual_strokes_per_block.keys():

        pt.img_batch = utils.img2patches(pt.img_, pt.m_grid, pt.net_G.out_size).to(device)
        pt.G_final_pred_canvas = CANVAS_tmp

        pt.manual_set_number_strokes_per_block(pt.m_grid)
        pt.initialize_params()
        pt.x_ctt.requires_grad = True
        pt.x_color.requires_grad = True
        pt.x_alpha.requires_grad = True
        utils.set_requires_grad(pt.net_G, False)

        pt.optimizer_x = optim.RMSprop([pt.x_ctt, pt.x_color, pt.x_alpha], lr=pt.lr, centered=True)

        pt.step_id = 0
        for pt.anchor_id in range(0, pt.m_strokes_per_block):
            pt.stroke_sampler(pt.anchor_id)
            iters_per_stroke = int(500 / pt.m_strokes_per_block)
            for i in range(iters_per_stroke):
                pt.G_pred_canvas = CANVAS_tmp

                # update x
                pt.optimizer_x.zero_grad()

                pt.clamp(val=clamp_schedule[pt.m_grid])

                pt._forward_pass()
                pt._drawing_step_states()
                pt._backward_x()

                pt.clamp(val=clamp_schedule[pt.m_grid])

                pt.optimizer_x.step()
                pt.step_id += 1

        v = pt._normalize_strokes(pt.x)
        v, idx_grid = pt._shuffle_strokes_and_reshape(v)

        # Add layer information
        layer_info = np.full((1, v.shape[1], 1), pt.m_grid)
        grid_info = np.repeat(idx_grid, pt.m_strokes_per_block)[None, :, None]    # repeat for each storke, add dim 0 and -1
        v = np.concatenate([v, layer_info, grid_info], axis=-1)

        # Add on previous parmas
        PARAMS = np.concatenate([PARAMS, v], axis=1)
        CANVAS_tmp, _ = pt._render(PARAMS, save_jpgs=False, save_video=False)
        CANVAS_tmp = utils.img2patches(CANVAS_tmp, pt.m_grid + 1, pt.net_G.out_size).to(device)

    PARAMS = pt.get_checked_strokes(PARAMS)
    pt._save_stroke_params(PARAMS)
    final_rendered_image, alphas = pt._render(PARAMS, save_jpgs=False, save_video=True)

    return final_rendered_image, alphas, PARAMS


def get_args():
    # settings
    parser = argparse.ArgumentParser(description='STYLIZED NEURAL PAINTING')
    parser.add_argument('--data_path', default='/home/eperuzzo/ade20k_outdoors/images/training/')
    parser.add_argument('--annotations_path', default='./annotations/training/')
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
    parser.add_argument('--renderer_checkpoint_dir', default='/home/eperuzzo/checkpoints_G_oilpaintbrush', type=str)
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--output_dir', default='./original_algorithm', type=str)
    parser.add_argument('--disable_preview', default=True, type=bool)
    parser.add_argument('--clamp_w_h', default=0.9, type=float)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--plot_losses', default=True, type=bool)
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
        'ADE_train_00001487.jpg']
    # [f for f in os.listdir(args.data_path) if f.endswith(('.jpg', '.png'))]
    k = 1
    for source_image in source_images:
        start = datetime.now()
        print('Processing image: {}, {}/{}'.format(source_image, k, len(source_images)))
        k += 1
        args.img_path = os.path.join(args.data_path, source_image)
        print(args.img_path)
        pt = NewPainter(args=args)
        final_rendered_image, alpha_transparency, strokes = optimize_x(pt, args.clamp_w_h)

        ## Save loss curves dictonary and figures
        elapsed = datetime.now() - start
        logs = pt.loss_dict
        logs['elapsed_time'] = elapsed
        with open(os.path.join(pt.output_dir, 'logs.pkl'), 'wb') as f:
            pickle.dump(logs, f, pickle.HIGHEST_PROTOCOL)
        if args.plot_losses:
            for elem, vals in logs.items():
                if elem == 'elapsed_time':
                    continue
                f = plt.figure(figsize=(10, 10))
                plt.plot(vals, color='green')
                plt.xlabel('iters')
                plt.ylabel(elem)
                plt.savefig(os.path.join(pt.output_dir, elem + '.png'))
                plt.close(f)

