import argparse
import os
from torch.utils.data import DataLoader
from model.model import InteractivePainter
from model.utils.parse_config import ConfigParser
from model.utils.utils import dict_to_device

import torch
from dataset_acquisition.decomposition.painter import Painter
from dataset_acquisition.decomposition.utils import load_painter_config
import matplotlib.pyplot as plt
from model.dataset import StrokesDataset, GenerativeDataset
import torchvision.transforms as transforms

from model.baseline.model import PaintTransformer


def unnormalize(x, resize=False) :
    if resize :
        t = transforms.Resize((512, 512))
        x = t(x)

    x = x.permute(1, 2, 0).detach().cpu().clone()

    for i in range(3) :
        x[:, :, i] = (x[:, :, i] * 0.5) + 0.5

    return x

def render_and_save(strokes, canvas_start, painter, name, output_path = '.'):

    res, _ = painter.inference(strokes)
    plt.imsave(os.path.join(output_path, name + '_strokes.png'), res)

    res, _ = painter.inference(strokes, canvas_start=canvas_start)
    plt.imsave(os.path.join(output_path, name + '_canvas.png'), res)


if __name__ == '__main__':
    # Extra parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--config", default='/home/eperuzzo/brushstrokes-generation/configs/train/sibiu_config.yaml')
    args = parser.parse_args()


    c_parser = ConfigParser(args, isTrain=False)
    c_parser.parse_config(args)
    config = c_parser.get_config()

    # Interactive Painter (our)
    IP = InteractivePainter(config)
    ckpt = torch.load('/home/eperuzzo/checkpoint_brushstrokes/add_7.5e-5_oxford/latest.pth.tar', map_location='cpu')
    IP.load_state_dict(ckpt["model"])
    IP.to(config["device"])
    IP.eval()

    # PainterTransformer
    PT = PaintTransformer(model_path= '/home/eperuzzo/PaintTransformer/inference/paint_best.pdparams',
                          input_size=256)

    # Create renderer
    renderer = Painter(args=load_painter_config(config["render"]["painter_config"]))


    # Test
    dataset_test = GenerativeDataset(config, isTrain=False)

    # beagle_22, Siamese_170, yorkshire_terrier_22
    names = ['beagle_22', 'leonberger_4', 'Abyssinian_8']
    for img_name in names:
        data = dataset_test.laod_filename_time_step(img_name, 100)

        canvas_start = data['canvas_ctx'][0, -1].permute(1,2,0).numpy()
        canvas_start = (canvas_start * 0.5) + 0.5

        data = dict_to_device(data, device=config["device"], to_skip=['time_steps'])

        path = os.path.join('./results/', img_name)
        os.makedirs(path, exist_ok=True)
        # Baseline
        baseline_preds, dd = PT.main(data['img'], data['canvas_ctx'][:, -1], strokes_ctx=data['strokes_ctx'])
        render_and_save(baseline_preds, canvas_start, renderer, name='baseline', output_path=path)
        print(baseline_preds.shape)
        print(dd)

        # Our predictions
        for i in range(3):
            preds = IP.generate(data, no_z=True)# sample z form a (0,1) gaussian
            render_and_save(preds.detach().cpu().numpy(), canvas_start, renderer, name=f'ip_{i}', output_path=path)

        # Plot original reuslts
        render_and_save(data['strokes_seq'].detach().cpu().numpy(), canvas_start, renderer, name='original',  output_path=path)

        # Save input image
        img = data['img'][0].cpu().permute(1,2,0).numpy()
        img = (img * 0.5) + 0.5
        plt.imsave(os.path.join(path, 'ref_img.png'), img)

        # canvas start
        plt.imsave(os.path.join(path, 'start.png'), canvas_start)


