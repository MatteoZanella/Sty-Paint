import argparse
import os
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from einops import repeat
import torch.nn.functional as F

import sys
sys.path.insert(1, '../')

from model.utils.utils import dict_to_device
from model.utils.parse_config import ConfigParser
from model import build_model
from model.dataset import EvalDataset

from dataset_acquisition.decomposition.painter import Painter
from dataset_acquisition.decomposition.utils import load_painter_config

from evaluation.paint_transformer.model import PaintTransformer as PaddlePT
from evaluation.tools import produce_visuals


if __name__ == '__main__' :
    # Extra parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--checkpoint_baseline", type=str,
                        default='/home/eperuzzo/PaintTransformerPaddle/inference/paint_best.pdparams')

    parser.add_argument("--config", default='configs/eval/eval_oxford.yaml')
    parser.add_argument("--files_setup", default='/home/eperuzzo/config.csv')
    parser.add_argument("--output_path", type=str, default='./results/')
    args = parser.parse_args()

    # Create config
    c_parser = ConfigParser(args.config, isTrain=False)
    c_parser.parse_config()
    config = c_parser.get_config()
    print(config)

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

    # load checkpoint, update model config based on the stored config
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    config.update(dict(model=ckpt["config"]["model"]))

    model = build_model(config)
    msg = model.load_state_dict(ckpt["model"], strict=False)
    print(f'==> Loading model form {args.checkpoint}, with : {msg}')
    model.cuda()
    model.eval()

    # basline Paint Transformer
    baseline = PaddlePT(model_path=args.checkpoint_baseline, config=render_config)

    os.makedirs(args.output_path, exist_ok=True)
    files = pd.read_csv(args.files_setup)
    for idx, row in files.iterrows() :
        filename = row["filename"]
        ts = row["ts"]
        print(f'==> Processing img : {filename}, {ts}')

        batch, _, _ = dataset_test.sample(filename, ts)

        # batch =
        data = dict_to_device(batch, to_skip=['strokes', 'time_steps'])
        targets = data['strokes_seq']
        starting_point = batch['canvas'][0].permute(1, 2, 0).cpu().numpy()
        img = (batch['img'][0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')

        visuals = dict()
        our_predicitons = dict(
            our_wo_z=dict()
        )
        ## Our model
        n_iters = 20
        for n in range(n_iters):
            with torch.no_grad():
                if n == 0:
                    best = []
                    n_best = 1
                    for j in range(n_best):
                        preds = model.generate(data, n_samples=100, select_best=True)
                        best.append(preds["fake_data_random"])
                    best = torch.cat(best)
                    target = repeat(data['strokes_seq'], 'bs L n_params -> (bs n_samples) L n_params',
                                    n_samples=n_best)
                    score = torch.nn.functional.mse_loss(target[:, :, :4], best[:, :, :4],
                                                         reduction='none').mean(dim=[1,2])
                    idx = torch.argmin(score, dim=0)
                    fake_data_random = best[idx].unsqueeze(0)
                else:
                    preds = model.generate(data, n_samples=1, select_best=True)
                    fake_data_random = preds["fake_data_random"]
                our_predicitons['our_wo_z'].update({n : fake_data_random.cpu().numpy()})

        ## Baseline model
        baseline_predictions = baseline.generate(data)
        snp_predicitons = renderer.generate(data)
        snp2_predicitons = snp_plus.generate(data)

        if torch.is_tensor(baseline_predictions) :
            baseline_predictions = baseline_predictions.cpu().numpy()
            snp_predicitons = snp_predicitons.cpu().numpy()
            snp2_predicitons = snp2_predicitons.cpu().numpy()


        try:
            visuals.update({'pt' : produce_visuals(baseline_predictions,
                                                   renderer=renderer,
                                                   starting_canvas=starting_point,
                                                   ctx=batch['strokes_ctx'],
                                                   seq=batch['strokes_seq'])})

            visuals.update({'snp' : produce_visuals(snp_predicitons,
                                                    renderer=renderer,
                                                    starting_canvas=starting_point,
                                                    ctx=batch['strokes_ctx'],
                                                    seq=batch['strokes_seq'])})

            visuals.update({'snp2' : produce_visuals(snp2_predicitons,
                                                    renderer=renderer,
                                                    starting_canvas=starting_point,
                                                    ctx=batch['strokes_ctx'],
                                                    seq=batch['strokes_seq'])})

            visuals.update({'original' : produce_visuals(batch['strokes_seq'],
                                                         renderer=renderer,
                                                         starting_canvas=starting_point,)})
            visuals.update({'reference' : img})

            for name, tmp in our_predicitons.items():
                visuals.update({name : dict()})
                for n, preds in tmp.items():
                    if n == 0:
                        visuals[name].update({n : produce_visuals(preds,
                                                                  renderer=renderer,
                                                                  starting_canvas=starting_point,
                                                                  ctx=batch['strokes_ctx'],
                                                                  seq=batch['strokes_seq'])})
                    else:
                        visuals[name].update({n : produce_visuals(preds,
                                                                  renderer=renderer,
                                                                  starting_canvas=starting_point,
                                                                  ctx=batch['strokes_ctx'])})


            ext = 'png'
            plt.imsave(os.path.join(args.output_path, f'{filename}_reference_img.{ext}'), visuals['reference'])
            plt.imsave(os.path.join(args.output_path, f'{filename}_canvas.{ext}'), np.uint8(starting_point * 255))
            plt.imsave(os.path.join(args.output_path, f'{filename}_original_seq.{ext}'), visuals['original'])
            # Baselines
            plt.imsave(os.path.join(args.output_path, f'{filename}_pt_seq.{ext}'), visuals['pt'])
            plt.imsave(os.path.join(args.output_path, f'{filename}_snp_seq.{ext}'), visuals['snp'])
            plt.imsave(os.path.join(args.output_path, f'{filename}_snp2_seq.{ext}'), visuals['snp2'])

            for n in range(n_iters):
                plt.imsave(os.path.join(args.output_path, f'{filename}_our_seq_{str(n).zfill(2)}.{ext}'), visuals['our_wo_z'][n])
        except:
            print(f'Skipping: {filename}')

