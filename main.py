import argparse
from graph import Graph
import torch
torch.cuda.current_device()
import torch.optim as optim
# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import numpy as np
import utils
import torch
from collections import defaultdict
from new_painter_renderer import NewPainter
import os


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
        print('Processing: {}/{}'.format(i, n))

        curr = transparency[i]
        next_strokes = transparency[i + 1:]
        overlap = np.logical_and(curr, next_strokes).sum(axis=(1, 2)) / c_size ** 2
        adj_list[i] = np.nonzero(overlap > treshold)[0] + (i + 1)

    return adj_list


def optimize_x(pt):
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

                pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0.1, 1 - 0.1)
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
        v = pt._shuffle_strokes_and_reshape(v)  # Note remove the shuffle
        PARAMS = np.concatenate([PARAMS, v], axis=1)
        CANVAS_tmp, _ = pt._render(PARAMS, save_jpgs=False, save_video=False)
        CANVAS_tmp = utils.img2patches(CANVAS_tmp, pt.m_grid + 1, pt.net_G.out_size).to(device)

    pt._save_stroke_params(PARAMS)
    PARAMS = pt.get_checked_strokes(PARAMS)
    final_rendered_image, alphas = pt._render(PARAMS, save_jpgs=False, save_video=True)

    return final_rendered_image, alphas, PARAMS

def get_args():
    # settings
    parser = argparse.ArgumentParser(description='STYLIZED NEURAL PAINTING')
    args = parser.parse_args(args=[])
    args.img_path_base = './test_images'  # path to input photo
    args.img_path = None
    args.renderer = 'oilpaintbrush'  # [watercolor, markerpen, oilpaintbrush, rectangle]
    args.canvas_color = 'black'  # [black, white]
    args.canvas_size = 512  # size of the canvas for stroke rendering'
    args.keep_aspect_ratio = False  # whether to keep input aspect ratio when saving outputs
    args.max_m_strokes = 500  # max number of strokes
    args.max_divide = 5  # divide an image up-to max_divide x max_divide patches
    args.beta_L1 = 1.0  # weight for L1 loss
    args.with_ot_loss = False  # set True for imporving the convergence by using optimal transportation loss, but will slow-down the speed
    args.beta_ot = 0.1  # weight for optimal transportation loss
    args.net_G = 'zou-fusion-net'  # renderer architecture
    args.renderer_checkpoint_dir = './checkpoints_G_oilpaintbrush'  # dir to load the pretrained neu-renderer
    args.lr = 0.005  # learning rate for stroke searching
    args.output_dir = './imagenet_output'  # dir to save painting results
    args.disable_preview = True  # disable cv2.imshow, for running remotely without x-display

    return args

if __name__ == '__main__':

    args = get_args()
    source_images = [f for f in os.listdir(args.img_path_base) if f.endswith(('.jpg', '.png'))]

    for source_image in source_images:
        args.img_path = os.path.join(args.img_path_base, source_image)
        pt = NewPainter(args=args)
        final_rendered_image, alpha_transparency, strokes = optimize_x(pt)

        n = strokes.shape[1]
        adj_list = build_graph(alpha_transparency)

        assert n == len(adj_list)

        g = Graph(n, adj_list, strokes)

        idx_sort, score = g.sort()

        sort_final_result, _ = pt._render(strokes[:, idx_sort, :], 'sorted', save_video=True, save_jpgs=False)

        assert (sort_final_result == final_rendered_image).all()