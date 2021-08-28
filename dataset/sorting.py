import argparse
import cv2
import torch
import os
from datetime import datetime
import subprocess

from sorting.graph import Graph, GraphBuilder, dfs_paths
import sorting.utils as utils
from decomposition.painter import Painter
from decomposition.utils import load_painter_config

def get_args():
    # settings
    parser = argparse.ArgumentParser(description='STYLIZED NEURAL PAINTING')
    parser.add_argument('--data_path', default='/home/eperuzzo/brushstrokes/strokes_params/')
    parser.add_argument('--output_path', default='/home/eperuzzo/brushstrokes/comparison_results/')
    parser.add_argument('--imgs_path', default='/home/eperuzzo/ade20k_outdoors/images/training/')
    parser.add_argument('--annotations_path', default='/home/eperuzzo/ade20k_outdoors/annotations/training/')
    parser.add_argument('--gpu_id', default=0, type=int)
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

    painter_config = load_painter_config(args.painter_config)
    pt = Painter(args=painter_config)

    for source_image in source_images:
        print('Processing image: {}, {}/{}'.format(source_image, k, len(source_images)))
        k += 1
        output_dir = os.path.join(args.output_path, source_image)  # used by painter to save
        utils.make_dir_tree(output_dir)

        # --------------------------------------------------------------------------------------------------------------
        # Load images and add info about segmentaiton and saliency of different stokes
        args.img_path = os.path.join(args.imgs_path, source_image + '.jpg' )
        src_path = os.path.join(args.data_path, source_image)
        strokes_loader = utils.StrokesLoader(path=src_path)
        strokes, layer = strokes_loader.load_strokes()

        print(strokes.shape)
        _, alphas = pt.inference(strokes)

        annotation_path = os.path.join(args.annotations_path, source_image + '.png')
        sm = utils.load_segmentation(annotation_path, args.canvas_size)
        saliency = utils.extract_salient(args.img_path, args.canvas_size)
        graph_features = strokes_loader.add_segmentation_saliency(sm, saliency, args.canvas_size)

        cv2.imwrite(os.path.join(output_dir, 'saliency_map.png'), saliency * 255)
        #cv2.imwrite(os.path.join(args.output_dir, 'segmentation_map.png'), sm * 255)
        # --------------------------------------------------------------------------------------------------------------
        # Create the adj list and add precedence based on layers
        print('Building graph ...')
        gb = GraphBuilder(alphas, 0)
        gb.build_graph()

        adj_list = gb.get_adjlist(hidden=True)
        adj_list = gb.layer_precedence(adj_list, layer)

        adj_list = dfs_paths(adj_list)
        utils.save_pickle(adj_list,
                         path = os.path.join(output_dir, 'adjlist'))
        # -----------------------------------------------------------------------------
        # Build the graph and order with the greedy policy
        print('***   Greedy policy   ***')
        graph = Graph(adj_list, graph_features)

        weights = [
            {'color': 1, 'area': 1, 'pos': 0, 'class': 5, 'sal': 0},
            {'color': 1, 'area': 1, 'pos': 0, 'class': 5, 'sal': 2.5}]
            #{'color': 1, 'area': 1, 'pos': 0, 'class': 5, 'sal': 5},
            #{'color': 1, 'area': 1, 'pos': 2, 'class': 0, 'sal': 0}]


        for w in weights:
            start_time = datetime.now()
            graph.reset_weights()
            graph.set_weights(w)

            name = "greedy_" + f"col{w['color']}_area{w['area']}_pos{w['pos']}_cl{w['class']}_sal{w['sal']}"
            idx = graph.sort()
            _ = pt._render(strokes[:, idx, :], path= os.path.join(output_dir, 'greedy', 'videos', name),
                           save_video=True, save_jpgs=False)

            utils.save_pickle(idx,
                             path=os.path.join(output_dir, 'greedy', 'index', name))

            logs = utils.check_tour(graph, idx)
            time = datetime.now() - start_time

            with open(os.path.join(output_dir, 'greedy', 'log_' + name + '.txt'), 'w') as file:
                file.write(logs)
                file.write('Elapsed time: %s' % time)
        # --------------------------------------------------------------------------------------------------------------
        # Run LKH solver
        print('*'*40)
        print('LKH policy')
        path_lkh_files = os.path.join(output_dir, 'lkh', 'lkh_files')
        graph = Graph(adj_list, graph_features)

        for w in weights:
            start_time = datetime.now()
            graph.reset_weights()
            graph.set_weights(w)
            start = graph.starting_node()

            # Cost matrix
            C = utils.lkh_cost_matrix(graph, start)

            # Creat the configuration file
            name = "lkh_" + f"col{w['color']}_area{w['area']}_pos{w['pos']}_cl{w['class']}_sal{w['sal']}"

            lkh_config = utils.LKHConfig(default_config_path='sorting/lkh_configuration',
                                         name=name,
                                         num_nodes=graph.n_nodes+1,
                                         output_path=os.path.join(output_dir, 'lkh', 'lkh_files'))
            lkh_config.parse_files(cost_matrix=C)
            # -------------------------------------------------------
            # Run LKH
            cmd = [args.lkh_solver, lkh_config.conf_file_path]
            x = subprocess.run(cmd, capture_output=True, text=True)
            with open(lkh_config.output_path + '_stout.txt', 'w') as f:
                f.write(x.stdout.strip("\n"))

            # Open the file just saved restore the order and save the indexes
            with open(lkh_config.conf_file['TOUR_FILE'], 'r') as f:
                sol = f.readlines()
                idx = [int(i) - 1 for i in sol[6:-3]]   # 1 based index

            # Save
            _ = pt._render(strokes[:, idx, :], path=os.path.join(output_dir, 'lkh', 'videos', name), save_video=True, save_jpgs=False)

            utils.save_pickle(idx,
                             path=os.path.join(output_dir, 'lkh', 'index', name))


            logs = utils.check_tour(graph, idx)
            time = datetime.now() - start_time

            with open(os.path.join(output_dir, 'lkh', 'log_' + name + '.txt'), 'w') as file:
                file.write(logs)
                file.write('\nElapsed time: %s' % time)