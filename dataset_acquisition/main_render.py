import argparse
import os

from decomposition.painter import Painter
import torch
from decomposition.utils import load_painter_config
from sorting.utils import StrokesLoader
import shutil
import glob
import pickle

def get_args():
    # settings
    parser = argparse.ArgumentParser(description='STYLIZED NEURAL PAINTING')
    parser.add_argument('--dataset_path', required=True, help='where the dataset_acquisition will be stored')
    parser.add_argument('--index_path', default='/data/eperuzzo/brushstrokes-ade/brushstrokes-sorting/', help='base folder with sorting results')
    parser.add_argument('--strokes_path', default='/data/eperuzzo/brushstrokes-ade/brushstrokes-decomposition/', help='base folder with decomposition results')

    parser.add_argument('--images_path', default='/data/eperuzzo/ade20k_outdoors/images/training/')
    parser.add_argument('--painter_config', default='./decomposition/painter_config.yaml')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    # Decide which device we want to run on
    device = torch.device('cuda:0')

    # Create basedirectory
    os.makedirs(args.dataset_path)

    # Define Painter
    painter_config = load_painter_config(args.painter_config)
    pt = Painter(args=painter_config)

    assert os.listdir(args.strokes_path) == os.listdir(args.index_path)
    images_to_process = os.listdir(args.strokes_path)


    for img_name in images_to_process:
        print(f'Processing image {img_name}')
        tmp_path = os.path.join(args.dataset_path, img_name)
        os.mkdir(tmp_path)

        # Copy stuff
        shutil.copy(src=os.path.join(args.images_path, img_name + '.jpg'),
                    dst=os.path.join(tmp_path, img_name + '.jpg'))

        shutil.copy(src=os.path.join(args.strokes_path, img_name, 'strokes_params.npz'),
                    dst=os.path.join(tmp_path, 'strokes_params.npz'))


        # Render images with associate heuristic
        strokes_loader = StrokesLoader(path=os.path.join(args.strokes_path, img_name))
        strokes, layer = strokes_loader.load_strokes()

        idx_paths = glob.glob(os.path.join(args.index_path, img_name, 'lkh', 'index', '*.pkl'))

        for idx_path in idx_paths:
            name = os.path.basename(idx_path).split('.')[0]
            os.mkdir(os.path.join(tmp_path, f'render_{name}'))

            with open(idx_path, 'rb') as f:
                idx = pickle.load(f)

            pt.inference(strokes, order=idx,
                         output_path=os.path.join(tmp_path, f'render_{name}'),
                         save_video=False,
                         save_jpgs=True)

            shutil.copy(src=os.path.join(idx_path),
                        dst=os.path.join(tmp_path, name + '.pkl'))