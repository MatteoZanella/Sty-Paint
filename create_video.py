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
from collections import OrderedDict
import matplotlib.pyplot as plt
import PIL.Image as Image
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore")

from einops import rearrange, repeat
import torch.nn.functional as F

def count_parameters(net) :
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def sample_color(params, imgs) :
    if not torch.is_tensor(params):
        params = torch.tensor(params)
    bs, n_strokes, _ = params.shape
    img_temp = repeat(imgs, 'bs ch h w -> (bs L) ch h w', L=n_strokes)
    grid = rearrange(params[:, :, :2], 'bs L p -> (bs L) 1 1 p')
    color = F.grid_sample(img_temp, 2 * grid - 1, align_corners=False)
    color = rearrange(color, '(bs L) ch 1 1 -> bs L ch', L=n_strokes)
    color = color.repeat(1,1,2)
    out_params = torch.cat((params.clone()[:, :, :5], color), dim=-1)
    return out_params.cpu().numpy()


def render_frames(params, batch, renderer) :
    bs = params.shape[0]
    L = params.shape[1]
    frames = np.empty([bs, L, 256, 256, 3])
    alphas = np.empty([bs, L, 256, 256, 1])
    for i in range(bs) :
        x = batch['canvas'][i].permute(1, 2, 0).cpu().numpy()
        for l in range(L) :
            if (params[i, l, 2] <= 0.025) or (params[i, l, 3] <= 0.025) :
                params[i, l, 2] = 0.026
                params[i, l, 3] = 0.026

            x, alpha = renderer.inference(params[i, l, :][None, None, :], canvas_start=x)
            frames[i, l] = x
            alphas[i, l] = alpha[0, :, :, None]

    return dict(frames=frames, alphas=alphas)


def render_lpips(inp, renderer, batch, bs, n_samples) :
    out = dict()
    for name in inp.keys() :
        out[name] = np.empty([bs, n_samples, 256, 256, 3])

        for b in range(bs) :
            for n in range(n_samples) :
                cs = batch['canvas'][b].permute(1, 2, 0).cpu().numpy()
                out[name][b, n :, :, :] = \
                renderer.inference(inp[name][n][b, :, :][None].cpu().numpy(), canvas_start=cs)[0]

    return out

def prepare_feature_difference(preds_lpips, bs, n_samples):
    out = dict()
    for key in preds_lpips.keys():
        out[key] = torch.empty((bs, n_samples, 8, 11))
        for n in range(n_samples) :
            out[key][:, n, :, :] = preds_lpips[key][n]

    return out


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
    parser.add_argument("--config", default='/home/eperuzzo/brushstrokes-generation/configs/train/sibiu_config.yaml')

    parser.add_argument("--output_path", type=str, default='/home/eperuzzo/eval_metrics/')
    parser.add_argument("--checkpoint_baseline", type=str,
                        default='/home/eperuzzo/PaintTransformer/inference/paint_best.pdparams')
    parser.add_argument("--n_samples_lpips", type=int, default=3,
                        help="number of samples to test lpips, diverstiy in generation")
    parser.add_argument("--n_iters_dataloader", default=1, type=int)
    args = parser.parse_args()


    # Create config
    c_parser = ConfigParser(args, isTrain=False)
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

    files = {
        'Abyssinian_206' : 50}

    os.makedirs(args.output_path, exist_ok=True)

    T =  transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(), ])



    batch = dataset_test.sample( 'Abyssinian_206', 50)
    ref_img = batch['img']
    starting_point = batch['canvas'][0].permute(1, 2, 0).cpu().numpy()

    n_iters = 30
    predictions = dict()
    results = dict()

    plt.imsave(os.path.join(args.output_path, f'frame_0.png'), starting_point)
    for n in range(n_iters):

        data = dict_to_device(batch, device, to_skip=['strokes', 'time_steps'])
        preds = net.generate(data)
        this_frame, _ = renderer.inference(preds.cpu().numpy(), canvas_start=starting_point)
        plt.imsave(os.path.join(args.output_path, f'frame_{n+1}.png'), this_frame)
        starting_point = this_frame
        # Update batch
        batch = {
            'img' : ref_img,
            'canvas' : torch.tensor((this_frame)).unsqueeze(0).permute(0, 3, 1, 2),
            'strokes_ctx' : preds,
            'strokes_seq' : torch.randn([1, 8, 11])
        }

