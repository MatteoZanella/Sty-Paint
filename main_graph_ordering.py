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


def clear_adj(x, hidden=False):
    '''
    hidden = True remove the to_remove nodes from the adj list
    '''
    out = defaultdict(list)

    for k, elem in x.items():
        if hidden:
            out[k] = [e for e in elem['all_edges'] if e not in elem['to_remove']]
        else:
            out[k] = [e for e in elem['all_edges']]

    return out

def assing_segm_label(graph, seg_map):
    # Assing a class to each stroke
    for i in range(graph.n_nodes):
        x0, y0 = graph.nodes[i].position
        x0 = _normalize(x0, args.canvas_size)
        y0 = _normalize(y0, args.canvas_size)
        graph.nodes[i].cl = seg_map[y0, x0]

def load_strokes(path):
    path = os.path.join(path, 'strokes_params.npz')

    x_ctt = np.load(path)['x_ctt']
    x_color = np.load(path)['x_color']
    x_alpha = np.load(path)['x_alpha']
    x_layer = np.load(path)['x_layer']

    strokes = np.concatenate([x_ctt, x_color, x_alpha], axis=-1)

    return strokes, x_layer


def _normalize(x, width):
    return (int)(x * (width - 1) + 0.5)


def get_args():
    # settings
    parser = argparse.ArgumentParser(description='STYLIZED NEURAL PAINTING')
    parser.add_argument('--data_path', default='/home/eperuzzo/brushstorkes/pippo/')
    parser.add_argument('--imgs_path', default='/home/eperuzzo/ade20k_outdoors/images/training/')
    parser.add_argument('--annotations_path', default='/home/eperuzzo/ade20k_outdoors/annotations/training/')
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
    parser.add_argument('--output_dir', default='', type=str)
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

    source_images = os.listdir(args.data_path)
    k = 1
    for source_image in source_images:
        print('Processing image: {}, {}/{}'.format(source_image, k, len(source_images)))
        k += 1
        args.output_dir = os.path.join(args.data_path, source_image)
        args.img_path = os.path.join(args.imgs_path, source_image + '.jpg' )
        logs = {}
        start0 = datetime.now()
        strokes, layer_info = load_strokes(os.path.join(args.data_path, source_image))
        pt = NewPainter(args=args)
        _, alphas = pt._render(strokes, '', save_video=False, save_jpgs=False)  # no needed later

        annotation_path = os.path.join(args.annotations_path, source_image + '.png')
        sm = load_segmentation(annotation_path, args.canvas_size)

        ## start form here
        print('Building graph ...')
        start = datetime.now()
        gb = GraphBuilder(alphas, 0)
        adj = gb.build_graph()
        adj_list = clear_adj(adj, hidden=True)  # remove unimportnat overlaps

        # Without layer info
        graph = Graph(adj_list, strokes)
        assing_segm_label(graph, sm)
        idx = graph.sort()

        _ = pt._render(strokes[:, idx, :], 'segmentation', save_video=True, save_jpgs=False)  # no needed later

        # Add layer information
        id_first = list(np.nonzero(layer_info[0, :, 0] == 2)[0])
        id_second = list(np.nonzero(layer_info[0, :, 0] != 2)[0])
        adj_list_layer = clear_adj(adj, hidden=True)  # pull the list again
        for ii in id_first:
            s = adj_list_layer[ii]
            s.extend(x for x in id_second if x not in s)
            s.sort()
            adj_list_layer[ii] = s

        graph_layer = Graph(adj_list_layer, strokes)
        assing_segm_label(graph, sm)
        idx_layer = graph_layer.sort()

        _ = pt._render(strokes[:, idx_layer, :], 'segmentation_layer', save_video=True, save_jpgs=False)  # no needed later

        elapsed = datetime.now() - start
        print('--> Elapsed time: {}'.format(elapsed))
        print('='*30)
        logs['elapsed_time'] = elapsed
        with open(os.path.join(pt.output_dir, 'graph_logs.pkl'), 'wb') as f:
            pickle.dump(logs, f, pickle.HIGHEST_PROTOCOL)