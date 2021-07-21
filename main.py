import argparse
from graph import Graph
import torch
torch.cuda.current_device()
import torch.optim as optim

import numpy as np
import utils
import torch
from collections import defaultdict
from new_painter_renderer import NewPainter
import os
import pickle


def check_storkes(s, renderer):
    s = s[0, :, :]

    good = []
    for i in range(s.shape[0]):
        renderer.stroke_params = s[i, :]
        if renderer.check_stroke():
            good.append(s[i, :][None, :])
    return np.concatenate(good, axis=0)[None, :, :]


def build_graph(transparency):
    n = transparency.shape[0]
    c_size = transparency[0].shape[0]
    treshold = 0

    adj_list = defaultdict(list)
    for i in range(n):
        curr = transparency[i]
        next_strokes = transparency[i + 1:]
        overlap = np.logical_and(curr, next_strokes).sum(axis=(1, 2)) / c_size ** 2
        adj_list[i] = np.nonzero(overlap > treshold)[0] + (i + 1)

    return adj_list


def optimize_x(pt, clamp_w_h):
    pt._load_checkpoint()
    pt.net_G.eval()
    pt._make_output_dir()

    print('begin drawing...')

    PARAMS = np.zeros([1, 0, pt.rderr.d], np.float32)

    if pt.rderr.canvas_color == 'white':
        CANVAS_tmp = torch.ones([1, 3, 128, 128]).to(device)
    else:
        CANVAS_tmp = torch.zeros([1, 3, 128, 128]).to(device)

    for pt.m_grid in range(1, pt.max_divide + 1):

        pt.img_batch = utils.img2patches(pt.img_, pt.m_grid, pt.net_G.out_size).to(device)
        pt.G_final_pred_canvas = CANVAS_tmp

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

                # Modification, use a different clamp for width and height
                pos = torch.clamp(pt.x_ctt.data[:, :, :2], 0.1, 1 - 0.1)
                size = torch.clamp(pt.x_ctt.data[:, :, 2:4], 0.1, clamp_w_h)
                theta = torch.clamp(pt.x_ctt.data[:, :, 4], 0.1, 1 - 0.1)

                # Put all back together
                pt.x_ctt.data = torch.cat([pos, size, theta.unsqueeze(-1)], dim=-1)
                pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
                pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 0, 1)

                pt._forward_pass()
                pt._drawing_step_states()
                pt._backward_x()

                pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0.1, 1 - 0.1)
                pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
                pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 0, 1)

                pt.optimizer_x.step()
                pt.step_id += 1

        v = pt._normalize_strokes(pt.x)
        v = pt._shuffle_strokes_and_reshape(v)
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
    parser.add_argument('--data_path', default='./test_images')
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
    #parser.add_argument('--alpha_spacing', default=0.2)
    return parser.parse_args()

if __name__ == '__main__':

    args = get_args()

    # Decide which device we want to run on
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu_id))
    else:
        device = torch.device('cpu')

    source_images = [f for f in os.listdir(args.data_path) if f.endswith(('.jpg', '.png'))]
    alphas = [0, 0.2, 0.4, 0.6, 0.8, 1]
    k = 1
    for source_image in source_images:
        print('Processing image: {}, {}/{}'.format(source_image, k, len(source_images)))
        k += 1
        args.img_path = os.path.join(args.data_path, source_image)
        pt = NewPainter(args=args)
        final_rendered_image, alpha_transparency, strokes = optimize_x(pt, args.clamp_w_h)

        n = strokes.shape[1]
        adj_list = build_graph(alpha_transparency)

        assert n == len(adj_list)
        g = Graph(n, adj_list, strokes)

        for alpha in alphas:
            idx_sort, score = g.sort(alpha)
            sort_final_result, _ = pt._render(strokes[:, idx_sort, :], 'sorted_lam_{}'.format(alpha), save_video=True, save_jpgs=False)
            assert (sort_final_result == final_rendered_image).all()
            with open(os.path.join(args.output_path, source_image[:-3], 'idx_lam_{}.pkl'.format(alpha)), 'wb') as f:
                pickle.dump(idx_sort, f)