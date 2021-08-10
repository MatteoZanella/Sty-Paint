import argparse
from graph_saliency import Graph, GraphBuilder
import cv2
import torch

torch.cuda.current_device()
import numpy as np
import torch
from new_painter_renderer import NewPainter
import os

from graph_utils import clear_adj, dfs_paths, lkh_cost_matrix
import subprocess
import pickle

def load_segmentation(path, cw):
    sm = cv2.imread(path)
    sm = cv2.resize(sm, (cw, cw), interpolation=cv2.INTER_NEAREST)
    sm = sm[:, :, 0]

    return sm

def extract_salient(img_path, size):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (size, size))

    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(img)
    sal = saliencyMap > 0.75 * saliencyMap.mean()

    return sal

def add_segmentation_saliency(strokes, seg_map, sal_map, size):
    n = strokes.shape[1]
    segm_info = np.zeros((1, n, 1))
    sal_info = np.zeros((1, n, 1))

    # Assing a class to each stroke
    for i in range(n):
        x0, y0 = strokes[0, i, :2]
        x0 = _normalize(x0, size)
        y0 = _normalize(y0, size)

        segm_info[0, i, 0] = seg_map[y0, x0]
        sal_info[0, i, 0] = sal_map[y0, x0]

    return np.concatenate([strokes, segm_info, sal_info], axis=-1)

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
    parser.add_argument('--data_path', default='/home/eperuzzo/brushstrokes/strokes_params/')
    parser.add_argument('--output_path', default='/home/eperuzzo/brushstrokes/comparison_results/')
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
    parser.add_argument('--output_dir', default='./original_algorithm', type=str)
    parser.add_argument('--disable_preview', default=True, type=bool)
    parser.add_argument('--clamp_w_h', default=0.9, type=float)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--plot_losses', default=True, type=bool)
    parser.add_argument('--lkh_solver', default='/home/eperuzzo/brushstrokes/solver/LKH-3.0.6/LKH', type=str)
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
        args.output_dir = os.path.join(args.output_path, source_image)  # used by painter to save
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        # --------------------------------------------------------------------------------------------------------------
        # Load images and add info about segmentaiton and saliency of different stokes
        args.img_path = os.path.join(args.imgs_path, source_image + '.jpg' )
        strokes, layer_info = load_strokes(os.path.join(args.data_path, source_image))
        pt = NewPainter(args=args)
        _, alphas = pt._render(strokes, '', save_video=False, save_jpgs=False)  # no needed later

        annotation_path = os.path.join(args.annotations_path, source_image + '.png')
        sm = load_segmentation(annotation_path, args.canvas_size)
        saliency = extract_salient(args.img_path, args.canvas_size)
        graph_features = add_segmentation_saliency(strokes, sm, saliency, args.canvas_size)

        cv2.imwrite(os.path.join(args.output_dir, 'saliency_map.png'), saliency * 255)
        #cv2.imwrite(os.path.join(args.output_dir, 'segmentation_map.png'), sm * 255)
        # --------------------------------------------------------------------------------------------------------------
        # Create the adj list and add precedence based on layers
        print('Building graph ...')
        gb = GraphBuilder(alphas, 0)
        adj = gb.build_graph()

        # Add layer information
        id_first = list(np.nonzero(layer_info[0, :, 0] == 2)[0])
        id_second = list(np.nonzero(layer_info[0, :, 0] != 2)[0])
        adj_list = clear_adj(adj, hidden=False)  # list, without unimportant overlaps
        for ii in id_first:
            s = adj_list[ii]
            s.extend(x for x in id_second if x not in s)
            s.sort()
            adj_list[ii] = s

        adj_list = dfs_paths(adj_list)
        with open(os.path.join(args.output_dir, 'adjlist.pkl'), 'wb') as f:
            pickle.dump(adj_list, f)
        # -------------------------------------------------------------------------------------------------------------
        # Build the graph and order with the greedy policy
        print('***   Greedy policy   ***')
        graph = Graph(adj_list, graph_features)

        weights = [
            {'color': 1, 'area': 1, 'pos': 0, 'class': 5, 'sal': 0},
            {'color': 1, 'area': 1, 'pos': 0, 'class': 5, 'sal': 5}]


        graph.reset_weights()
        for w in weights:
            graph.set_weights(w)
            idx = graph.sort()
            _ = pt._render(strokes[:, idx, :], 'greedy_cl_{}_sal_{}'.format(w['class'], w['sal']), save_video=True, save_jpgs=False)
            graph.reset_weights()

            with open(os.path.join(args.output_dir, 'idx_greedy_cl_{}_sal_{}.pkl'.format(w['class'], w['sal'])), 'wb') as f:
                pickle.dump(idx, f)

        # --------------------------------------------------------------------------------------------------------------
        # Run LKH solver
        print('***   LKH policy   ***')
        path_lkh_files = os.path.join(args.output_dir, 'lkh_files')
        graph = Graph(adj_list, graph_features)

        if not os.path.exists(path_lkh_files):
            os.makedirs(path_lkh_files)

        for w in weights:
            print('New iterations')
            graph.reset_weights()
            graph.set_weights(w)
            start = graph.starting_node()

            # Cost matrix
            C = lkh_cost_matrix(graph, start)

            # Creat the configuration file
            sop_name = 'lkh_cl_{}_sal_{}'.format(w['class'], w['sal'])
            with open('lkh_configuration/problem_file.pkl', 'rb') as f:
                problem_file = pickle.load(f)
            problem_file['NAME'] = ' {}.sop'.format(sop_name)
            problem_file['DIMENSION'] = graph.n_nodes+1
            problem_file['EDGE_WEIGHT_SECTION'] = graph.n_nodes + 1

            with open('lkh_configuration/conf_file.pkl', 'rb') as f:
                conf_file = pickle.load(f)
            conf_file['PROBLEM_FILE'] = os.path.join(path_lkh_files, sop_name + '.sop')
            conf_file['TOUR_FILE'] = os.path.join(path_lkh_files, sop_name + '_solution.txt')
            # -------------------------------------------
            # Write configuration files and run the LKH solver
            conf_file_path = os.path.join(path_lkh_files, 'config_' + sop_name + '.par')

            with open(os.path.join(path_lkh_files, sop_name + '.sop'), 'w') as f:
                for key, item in problem_file.items():
                    if key == 'EDGE_WEIGHT_SECTION':
                        f.write(key + '\n')
                        f.write(str(item))
                    else:
                        f.write(key + ': ' + str(item))
                    f.write('\n')

                # save the weight matrix
                np.savetxt(f, C, fmt='%d')
                f.write('\nEOF')

            with open(conf_file_path, 'w') as f:
                for key, item in conf_file.items():
                    f.write(key + '= ' + str(item))
                    f.write('\n')


            # -------------------------------------------------------
            # Run LKH
            cmd = [args.lkh_solver, conf_file_path]
            x = subprocess.run(cmd, capture_output=True, text=True)
            with open(conf_file_path[:-4] + 'stout.txt', 'w') as f:
                f.write(x.stdout.strip("\n"))

            # Open the file just saved restore the order and save the indexes
            with open(conf_file['TOUR_FILE'], 'r') as f:
                sol = f.readlines()

                ids = [int(i) for i in sol[6:-3]]
                ids = np.array(ids) - 1  # LKH is 1 based index

            # switch start with zero
            ids[ids==start] = 0
            ids[0] = start

            # Write it to pickle
            with open(os.path.join(args.output_dir, 'idx_' + sop_name + '.pkl'), 'wb') as f:
                pickle.dump(ids, f)

            _ = pt._render(strokes[:, ids, :], 'lkh_cl_{}_sal_{}'.format(w['class'], w['sal']), save_video=True,
                           save_jpgs=False)