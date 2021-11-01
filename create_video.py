import argparse
import os

from model.utils.utils import dict_to_device
from model.utils.parse_config import ConfigParser
from model import model, model_2_steps
from model.dataset import EvalDataset

from dataset_acquisition.decomposition.painter import Painter
from dataset_acquisition.decomposition.utils import load_painter_config
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import evaluation.tools as etools


import warnings
warnings.filterwarnings("ignore")


def count_parameters(net) :
    return sum(p.numel() for p in net.parameters() if p.requires_grad)



def produce_visuals(params, ctx, renderer, st) :
    fg, alpha = renderer.inference(params, canvas_start=st)
    _, alpha_ctx = renderer.inference(ctx)
    cont = visualize(fg, alpha, alpha_ctx)

    return cont


def visualize(foreground, alpha, alpha_ctx) :
    tmp = ((alpha.sum(0) * 255)[:, :, None]).astype('uint8')
    contours, hierarchy = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    tmp_alpha = ((alpha_ctx.sum(0) * 255)[:, :, None]).astype('uint8')
    contours_ctx, _ = cv2.findContours(tmp_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    x = (np.copy(foreground) * 255).astype('uint8')
    res = cv2.drawContours(x, contours_ctx, -1, (255, 0, 0), 1)
    res = cv2.drawContours(res, contours, -1, (0, 255, 0), 1)
    return res


if __name__ == '__main__' :
    # Extra parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_1", type=str, required=True)
    parser.add_argument("--no_z", action='store_false', help='set the flag to USE z')
    parser.add_argument("--config", default='/home/eperuzzo/brushstrokes-generation/configs/train/sibiu_config.yaml')
    parser.add_argument("--n_iters", default=30, type=int, help='Number of iterations to generate strokes')
    parser.add_argument("--output_path", type=str, default='/home/eperuzzo/eval_metrics/')
    parser.add_argument("--checkpoint_baseline", type=str,
                        default='/home/eperuzzo/PaintTransformer/inference/paint_best.pdparams')
    parser.add_argument("--n_samples_lpips", type=int, default=3,
                        help="number of samples to test lpips, diverstiy in generation")
    parser.add_argument("--n_iters_dataloader", default=1, type=int)
    args = parser.parse_args()


    # Create config
    c_parser = ConfigParser(args.config, isTrain=False)
    c_parser.parse_config(args)
    config = c_parser.get_config()
    print(config)

    # Create dataset_acquisition
    device = config["device"]

    # Test
    dataset_test = EvalDataset(config, isTrain=False)
    #test_loader = DataLoader(dataset=dataset_test, batch_size=16, shuffle=True, pin_memory=False)
    print(f'Test : {len(dataset_test)} samples')

    # ======= Create Models ========================
    # Renderer (Stylized Neural Painting)
    render_config = load_painter_config(config["render"]["painter_config"])
    renderer = Painter(args=render_config)

    net = model.InteractivePainter(config)
    net.load_state_dict(torch.load(args.ckpt_1, map_location=device)["model"])
    net.to(config["device"])
    net.eval()

    os.makedirs(args.output_path, exist_ok=True)
    T =  transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(), ])

    filename = 'Abyssinian_2'
    t_start = 50
    batch = dataset_test.sample(filename, t_start)
    ref_img = batch['img']
    starting_point = batch['canvas'][0].permute(1, 2, 0).cpu().numpy()

    n_iters = args.n_iters
    predictions = dict()
    results = dict()

    plt.imsave(os.path.join(args.output_path, f'frame_0.png'), starting_point)
    for n in range(n_iters):

        data = dict_to_device(batch, device, to_skip=['strokes', 'time_steps'])
        preds = net.generate(data, no_z=args.no_z)
        this_frame, _ = renderer.inference(preds.cpu().numpy(), canvas_start=starting_point)
        plt.imsave(os.path.join(args.output_path, f'frame_{n+1}.png'), this_frame)
        starting_point = this_frame
        # Update batch
        if args.no_z:
            batch = {
                'img' : ref_img,
                'canvas' : torch.tensor((this_frame)).unsqueeze(0).permute(0, 3, 1, 2),
                'strokes_ctx' : preds,
                'strokes_seq' : torch.randn([1, 8, 11])
            }
        else:
            # maybe we should put as context the generated strokes
            t_start += 8
            batch = dataset_test.sample(filename, t_start)
            batch['strokes_ctx'][:, -8:, :] = preds
