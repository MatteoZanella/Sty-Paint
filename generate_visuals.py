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
from model.dataset import StrokesDataset

from model.baseline.model import PaintTransformer


def unnormalize(x) :
    return (x * 0.5) + 0.5    # [-1, 1] to [0, 1]

def render_and_save(strokes, canvas_start, painter, name, output_path = '.'):

    res, _ = painter.inference(strokes)
    plt.imsave(os.path.join(output_path, name + '_strokes.png'), res)

    res, _ = painter.inference(strokes, canvas_start=canvas_start)
    plt.imsave(os.path.join(output_path, name + '_canvas.png'), res)


if __name__ == '__main__':
    # Extra parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--baseline_ckpt", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./results/")
    args = parser.parse_args()


    c_parser = ConfigParser(args, isTrain=False)
    c_parser.parse_config(args)
    config = c_parser.get_config()

    os.makedirs(args.output_path, exist_ok=True)
    # Renderer
    config_renderer = load_painter_config(config["render"]["painter_config"])
    renderer = Painter(args=config_renderer)

    # Interactive Painter (our)
    IP = InteractivePainter(config)
    ckpt = torch.load(args.model_ckpt, map_location='cpu')
    IP.load_state_dict(ckpt["model"])
    IP.to(config["device"])
    IP.eval()

    # PainterTransformer
    PT = PaintTransformer(model_path= args.baseline_ckpt,
                          config=config_renderer)



    # Test
    dataset_test = StrokesDataset(config, isTrain=False)
    dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=True)

    data = next(iter(dataloader))
    # Unnormalize
    canvas_start = data['canvas'][0].permute(1, 2, 0).numpy()
    canvas_start = unnormalize(canvas_start)

    data = dict_to_device(data, device=config["device"], to_skip=['time_steps'])

    # Baseline
    baseline_preds, _ = PT.main(data['img'], data['canvas'], strokes_ctx=data['strokes_ctx'])
    render_and_save(baseline_preds, canvas_start, renderer, name='baseline', output_path=args.output_path)

    # Our predictions
    for i in range(3):
        preds = IP.generate(data, no_z=True)    # sample z form a (0,1) gaussian
        render_and_save(preds.detach().cpu().numpy(), canvas_start, renderer, name=f'ip_{i}', output_path=args.output_path)

    # Plot original results
    render_and_save(data['strokes_seq'].detach().cpu().numpy(), canvas_start, renderer, name='original',  output_path=args.output_path)

    # Save input image
    img = data['img'][0].cpu().permute(1,2,0).numpy()
    img = unnormalize(img)
    plt.imsave(os.path.join(args.output_path, 'ref_img.png'), img)
    plt.imsave(os.path.join(args.output_path, 'start.png'), canvas_start)


