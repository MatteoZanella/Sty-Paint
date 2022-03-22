import argparse
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import copy

import sys
sys.path.insert(1, '../')

from model.utils.utils import dict_to_device
from model.utils.parse_config import ConfigParser
from model import build_model
from model.dataset import EvalDataset
from evaluation.paint_transformer.model import PaintTransformer as PaddlePT
from evaluation.tools import check_strokes

from dataset_acquisition.decomposition.painter import Painter
from dataset_acquisition.decomposition.utils import load_painter_config

from evaluation.tools import produce_visuals


def predict(net, batch, renderer, n_iters=100, is_our=True):
    data = dict_to_device(batch, to_skip=['strokes', 'time_steps'])

    alphas = []
    if is_our :
        num_samples = n_iters
        Z = torch.randn(num_samples, 256)
        with torch.no_grad() :
            fake_data_random = net.sample(data, Z.cuda())
            for ii in range(fake_data_random.shape[0]) :
                preds = fake_data_random[ii].unsqueeze(0).cpu()
                _, alpha = renderer.inference(preds)
                alphas.append(alpha)
    else:
        for n in range(n_iters):
            preds = net.generate(data)
            if not torch.is_tensor(preds):
                preds = torch.tensor(preds)
            preds = check_strokes(preds)
            _, alpha = renderer.inference(preds)
            alphas.append(alpha)

    alphas = np.concatenate(alphas).sum(axis=0)
    return alphas

if __name__ == '__main__' :
    # Extra parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config",
                        default='/home/eperuzzo/brushstrokes-generation/configs/eval/eval_oxford.yaml')
    parser.add_argument("--files_setup", default='/home/eperuzzo/config.csv')
    parser.add_argument("--output_path", type=str, default='./results/')
    parser.add_argument("--checkpoint_pt", type=str,
                        default='/home/eperuzzo/PaintTransformerPaddle/inference/paint_best.pdparams')
    parser.add_argument('--n_iters', default=500, type=int)
    args = parser.parse_args()


    # Create config
    c_parser = ConfigParser(args.config, isTrain=False)
    c_parser.parse_config()
    config = c_parser.get_config()

    # Test
    dataset_test = EvalDataset(config, isTrain=False)
    print(f'Test : {len(dataset_test)} samples')

    # ======= Create Models ========================
    # Renderer (Stylized Neural Painting)
    render_config = load_painter_config(config["renderer"]["painter_config"])
    renderer = Painter(args=render_config)

    snp_plus_config = copy.deepcopy(render_config)
    snp_plus_config.with_kl_loss = True
    snp_plus = Painter(args=snp_plus_config)

    pt = PaddlePT(model_path=args.checkpoint_pt, config=render_config)

    # load checkpoint, update model config based on the stored config
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    config.update(dict(model=ckpt["config"]["model"]))

    model = build_model(config)
    msg = model.load_state_dict(ckpt["model"], strict=False)
    print(f'==> Loading model form {args.checkpoint}, with : {msg}')
    model.cuda()
    model.eval()

    files = pd.read_csv(args.files_setup)
    for idx, row in files.iterrows() :
        filename = row["filename"]
        ts = row["ts"]
        print(f'==> Processing img : {filename}')
        os.makedirs(args.output_path, exist_ok=True)

        # Process
        batch, _, _ = dataset_test.sample(filename, ts)

        # batch =
        data = dict_to_device(batch, to_skip=['strokes', 'time_steps'])
        targets = data['strokes_seq']
        starting_point = batch['canvas'][0].permute(1, 2, 0).cpu().numpy()
        img = (batch['img'][0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')

        # Save context
        context_img = produce_visuals(params=None, ctx=batch['strokes_ctx'], renderer=renderer,
                                      starting_canvas=starting_point)

        plt.imsave(os.path.join(args.output_path,
                                f'{filename}_context.png'),
                   context_img)

        # Our
        alpha_our = predict(model, batch, renderer=renderer, n_iters=args.n_iters, is_our=True)
        plt.imsave(os.path.join(args.output_path,
                                f'{filename}_our.png'),
                   alpha_our)

        # PT
        alpha_pt = predict(pt, batch, renderer=renderer, n_iters=args.n_iters, is_our=False)
        plt.imsave(os.path.join(args.output_path,
                                f'{filename}_pt.png'),
                   alpha_pt)

        # SNP
        alpha_snp = predict(renderer, batch, renderer=renderer, n_iters=args.n_iters, is_our=False)
        plt.imsave(os.path.join(args.output_path,
                                f'{filename}_snp.png'),
                   alpha_snp)
        # SNP+
        alpha_snp2 = predict(snp_plus, batch, renderer=renderer, n_iters=args.n_iters, is_our=False)
        plt.imsave(os.path.join(args.output_path,
                                f'{filename}_snp2.png'),
                   alpha_snp2)