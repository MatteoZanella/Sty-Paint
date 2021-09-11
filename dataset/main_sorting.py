import argparse
import os
import shutil
from datetime import datetime
import subprocess

from sorting.graph import Graph, GraphBuilder, dfs_paths
import sorting.utils as utils
import queue
import threading

import numpy as np
import pandas as pd

def get_args():
    # settings
    parser = argparse.ArgumentParser(description='STYLIZED NEURAL PAINTING')
    parser.add_argument('--csv_file', default='/data/eperuzzo/brushstrokes-generation/code/dataset/chunks/todi.csv')
    parser.add_argument('--data_path', default='/data/eperuzzo/ade_v1/')
    parser.add_argument('--output_path', default='/data/eperuzzo/sorting_ade_v2/')
    parser.add_argument('--imgs_path', default='/data/eperuzzo/ade20k_outdoors/images/training/')
    parser.add_argument('--annotations_path', default='/data/eperuzzo/ade20k_outdoors/annotations/training/')
    parser.add_argument('--n_threads', default=20, type=int)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--lkh_solver', default='/data/eperuzzo/brushstrokes-generation/code/solver/LKH-3.0.6/LKH', type=str)
    parser.add_argument('--canvas_size', default=512, type=int)
    return parser.parse_args()

def process_image(args, source_image, w):
    print(source_image)

    output_dir = os.path.join(args.output_path, source_image)  # used by painter to save
    utils.make_dir_tree(output_dir)

    # --------------------------------------------------------------------------------------------------------------
    # Load images and add info about segmentaiton and saliency of different stokes
    args.img_path = os.path.join(args.imgs_path, source_image + '.jpg')
    src_path = os.path.join(args.data_path, source_image)
    strokes_loader = utils.StrokesLoader(path=src_path)
    strokes, layer = strokes_loader.load_strokes()

    annotation_path = os.path.join(args.annotations_path, source_image + '.png')
    sm = utils.load_segmentation(annotation_path, args.canvas_size)
    saliency = utils.extract_salient(args.img_path, args.canvas_size)
    graph_features = strokes_loader.add_segmentation_saliency(sm, saliency, args.canvas_size)
    # --------------------------------------------------------------------------------------------------------------
    # Create the adj list and add precedence based on layers
    print('Building graph ...')
    alphas = np.load(os.path.join(src_path, 'alpha.npz'))['alpha']
    gb = GraphBuilder(alphas, 0)
    gb.build_graph()

    adj_list = gb.get_adjlist(hidden=True)
    adj_list = gb.layer_precedence(adj_list, layer)

    adj_list = dfs_paths(adj_list)
    utils.save_pickle(adj_list,
                      path=os.path.join(output_dir, 'adjlist'))

    # --------------------------------------------------------------------------------------------------------------
    # Run LKH solver
    print('*' * 40)
    print('LKH policy')
    path_lkh_files = os.path.join(output_dir, 'lkh', 'lkh_files')
    graph = Graph(adj_list, graph_features)


    start_time = datetime.now()
    graph.reset_weights()
    graph.set_weights(w)
    start = graph.starting_node()

    # Cost matrix
    C = utils.lkh_cost_matrix(graph, start)

    # Creat the configuration file
    name = "lkh_" + f"col{w['color']}_area{w['area']}_pos{w['pos']}_cl{w['class']}_sal{w['sal']}"

    lkh_config = utils.LKHConfig(default_config_path='../configs/lkh_configuration',
                                 name=name,
                                 num_nodes=graph.n_nodes + 1,
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
        idx = [int(i) - 1 for i in sol[6:-3]]  # 1 based index

    utils.save_pickle(idx,
                      path=os.path.join(output_dir, 'lkh', 'index', name))

    logs = utils.check_tour(graph, idx)
    time = datetime.now() - start_time

    with open(os.path.join(output_dir, 'lkh', 'log_' + name + '.txt'), 'w') as file:
        file.write(logs)
        file.write('\nElapsed time: %s' % time)

    # Remove the problem file, it is too big
    print('Delete problem file ...')
    os.remove(f'{lkh_config.output_path}.sop')

def worker():
    while True:
        image, ww = q.get()
        try:
            process_image(args, image, ww)
        except:
            continue
        q.task_done()


if __name__ == '__main__':

    args = get_args()
    source_images = list(pd.read_csv(args.csv_file)['Images'])
    #source_images = os.listdir(args.data_path)

    q = queue.Queue()
    N_THREADS = args.n_threads


    for i in range(N_THREADS):
        # turn-on the worker thread
        threading.Thread(target=worker, daemon=True, name=str(i)).start()

    weights = [{'color': 2, 'area': 1, 'pos': 0, 'class': 7.5, 'sal': 0}]

    for source_image in source_images:
        for w in weights:
            q.put((source_image, w))
    print('All task requests sent')

    # block until all tasks are done
    q.join()
    print('All work completed')