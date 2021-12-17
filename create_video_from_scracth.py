import argparse
import os

from model.utils.utils import dict_to_device
from model.utils.parse_config import ConfigParser
from model import build_model
import PIL.Image as Image

from dataset_acquisition.decomposition.painter import Painter
from dataset_acquisition.decomposition.utils import load_painter_config
import torch
import torchvision.transforms as transforms
import evaluation.tools as etools
import numpy as np
import matplotlib.pyplot as plt
import glob
import warnings
warnings.filterwarnings("ignore")


def get_clamp_schedule(n, tot):
    # Clamp schedule
    v1 = np.ones(200)
    v2 = np.linspace(1, 0.4, 150)  # 350
    v3 = np.linspace(0.4, 0.1, 150) # 500
    v4 = np.linspace(0.1, 0.08, 200)  # 700
    v5 = np.linspace(0.08, 0.03, tot-700)

    vals = np.concatenate((v1, v2, v3, v4, v5))

    return vals[n]

def _to_tensor(x, normalize=True):
    if normalize:
        x = x / 255.0
    x = torch.tensor(x)
    return x.permute(2, 0, 1).unsqueeze(0)

if __name__ == '__main__' :
    # Extra parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--img_path", type=str, default='/home/eperuzzo/')

    parser.add_argument("--L", type=int, default=8, help='Number of strokes that will be predicted')
    parser.add_argument("--no_z", action='store_false', help='set the flag to USE z')
    parser.add_argument("--config", default='/home/eperuzzo/brushstrokes-generation/configs/eval/eval.yaml')
    parser.add_argument("--n_iters", default=100, type=int, help='Number of iterations to generate strokes')
    parser.add_argument("--output_path", type=str, default='/home/eperuzzo/our_video/')
    args = parser.parse_args()

    # Create config
    c_parser = ConfigParser(args.config, isTrain=False)
    c_parser.parse_config(args)
    config = c_parser.get_config()

    print(f'Sampling z : {args.no_z}, dimension of input {config["dataset"]["resize"]}')

    # Create dataset_acquisition
    device = config["device"]

    # params
    L = args.L
    ctx_len = config["dataset"]["context_length"]
    seq_len = config["dataset"]["sequence_length"]

    tot_strokes = args.L * args.n_iters
    print(f'Total number of strokes : {tot_strokes}')

    # ======= Create Models ========================
    # Renderer (Stylized Neural Painting)
    render_config = load_painter_config(config["renderer"]["painter_config"])
    renderer = Painter(args=render_config)

    model = build_model(config)
    print(f'==> Loading model form {args.checkpoint}')
    model.load_state_dict(torch.load(args.checkpoint)["model"])
    model.cuda()
    model.eval()

    os.makedirs(args.output_path, exist_ok=True)
    T =  transforms.Compose([
        transforms.Resize((config["dataset"]["resize"], config["dataset"]["resize"])),
        transforms.ToTensor(), ])

    # load image
    img_list = glob.glob(args.img_path + '*.jpg')
    output_path = os.path.join(args.output_path)
    os.makedirs(output_path, exist_ok=True)

    cumulative_loss = []
    for img_path in img_list:
        print(f'Processing image : {img_path}')
        ref_img = Image.open(img_path)
        ref_img = T(ref_img).unsqueeze(0)
        canvas_start = torch.zeros([1, 3, render_config.canvas_size, render_config.canvas_size], dtype=ref_img.dtype)
        strokes_ctx = torch.zeros([1, config["dataset"]["context_length"], config["model"]["n_strokes_params"]])
        strokes_seq = torch.zeros([1, config["dataset"]["sequence_length"], config["model"]["n_strokes_params"]])
        batch = {
            'img' : ref_img,
            'canvas' : canvas_start,
            'strokes_ctx' : strokes_ctx,
            'strokes_seq' : strokes_seq,}

        n_iters = args.n_iters
        predictions = dict()
        results = dict()

        starting_point = np.zeros((render_config.canvas_size, render_config.canvas_size, 3))
        strokes = []
        drawn_strokes = 0
        to_save = []
        for n in range(n_iters):
            clamp = get_clamp_schedule(drawn_strokes, tot_strokes)
            data = dict_to_device(batch, to_skip=['strokes', 'time_steps'])
            with torch.no_grad():
                preds = model(data, sample_z=True, seq_length=L)["fake_data_random"]
            preds = etools.check_strokes(preds, clamp_wh=clamp)
            strokes.append(preds.cpu().numpy())
            this_frame, _ = renderer.inference(preds.cpu().numpy(), canvas_start=starting_point)
            starting_point = this_frame

            to_save.append(this_frame)
            cumulative_loss.append(torch.nn.MSELoss()(_to_tensor(starting_point), batch['img']).item())

            # Update context
            ctx = batch['strokes_ctx'] # copy
            ctx = torch.roll(ctx, shifts=args.L, dims=1)   # shift the context by the number of predicted strokes
            ctx[:, :L, :] = torch.flip(preds, dims=(1,))[:, :ctx_len, :]  # FIFO

            batch = {
                'img' : ref_img,
                'canvas' : torch.tensor((this_frame), dtype=ref_img.dtype).unsqueeze(0).permute(0, 3, 1, 2),
                'strokes_ctx' : ctx,
                'strokes_seq' : torch.randn_like(preds)
            }
            drawn_strokes += args.L

        # for n in range(len(frames)):
        #     plt.imsave(os.path.join(args.output_path, f'frame_{n}.jpg'), frames[n])
        # frames = []
        #
        # print(strokes.shape)
        # for j in range(strokes.shape[1]):
        #     tmp, _ = renderer.inference(strokes[:, j, :][:, None, :])
        #     frames.append(tmp)

        strokes = np.concatenate(strokes, axis=1)
        img_name = os.path.basename(img_path).split('.')[0]
        out = os.path.join(args.output_path, img_name  + f'_L_{L}')
        renderer.inference(strokes, output_path=out, save_video=True)

        os.makedirs(os.path.join(args.output_path, img_name + '_renders'), exist_ok=True)

        for i in range(len(to_save)):
            plt.imsave(os.path.join(args.output_path, img_name + '_renders', f'frame_{str(i).zfill(3)}.jpg'), to_save[i])

        f = plt.figure()
        plt.plot(cumulative_loss)
        plt.savefig(os.path.join(args.output_path, img_name + '_loss.png'))
        #etools.create_video(frames, path=args.output_path, size=render_config.canvas_size, scale=True)