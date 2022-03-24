import argparse
import os
import pickle as pkl
import random
import numpy as np
import pandas as pd
import copy
import torch
import paddle
from torch.utils.data import DataLoader

from model.utils.utils import dict_to_device, AverageMetersDict
from model.utils.parse_config import ConfigParser
from model import build_model
from model.dataset import StrokesDataset

from dataset_acquisition.decomposition.painter import Painter
from dataset_acquisition.decomposition.utils import load_painter_config

from evaluation.metrics import LPIPSDiversityMetric, WassersteinDistance, FSD, StrokeColorL2, DTW, \
    compute_color_difference
from evaluation.fvd import FVD
from evaluation.paint_transformer.model import PaintTransformer as PaddlePT
import evaluation.tools as etools


class EvalModel:
    def __init__(self, renderer):
        super(EvalModel, self).__init__()
        self.renderer = renderer

        self.scl2Metric = StrokeColorL2()
        self.wdMetric = WassersteinDistance()
        self.dtwMetric = DTW()

    def __call__(self, data, net, metric_logger, is_our=False):

        targets = data['strokes_seq'].cpu()
        bs = targets.size(0)

        if is_our:
            predictions = net.generate(data)["fake_data_random"]
        else:
            predictions = net.generate(data)
        predictions = etools.check_strokes(predictions)  # clamp in range [0,1]

        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()

        visuals = etools.render_frames(predictions, data, self.renderer)

        # compute metrics
        color_diff_l1, color_diff_l2 = compute_color_difference(predictions)
        scl2 = self.scl2Metric(data['img'], visuals['frames'], visuals['alphas'])
        wd = self.wdMetric(targets[:, :, :5], torch.tensor(predictions[:, :, :5]))
        dtw = self.dtwMetric(targets, predictions)

        # record metrics
        metric_logger.update(
            dict(
                scl2=scl2.item(),
                wd=wd.item(),
                color_diff_l1=color_diff_l1.item(),
                color_diff_l2=color_diff_l2.item(),
                dtw=dtw.item()), bs)

        return predictions, visuals['frames']

def main(args):
    # Seed
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    paddle.seed(seed)
    torch.backends.cudnn.benchmark = True

    # Create config
    c_parser = ConfigParser(args.config, isTrain=False)
    c_parser.parse_config()
    config = c_parser.get_config()
    print(config)

    # Test
    dataset_test = StrokesDataset(config, isTrain=False)
    test_loader = DataLoader(dataset=dataset_test,
                             batch_size=64,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)
    print(f'Test : {len(dataset_test)} samples')

    # ======= Create Models ========================
    # Renderer (Stylized Neural Painting)
    render_config = load_painter_config(config["renderer"]["painter_config"])
    renderer = Painter(args=render_config)
    if args.use_snp2:
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

    # Output Path
    output_path = os.path.join(args.output_path, ckpt["config"]["train"]["logging"]["exp_name"])

    # Paint Transformer
    baseline = PaddlePT(model_path=args.pt_checkpoint, config=render_config)
    # ======= Metrics ========================
    fsdMetric = FSD()
    eval_model = EvalModel(renderer=renderer)
    original, ctx, our, pt, snp, snp2 = [], [], [], [], [], []
    visual_original, visual_ctx, visual_our, visual_pt, visual_snp, visual_snp2 = [], [], [], [], [], []

    # Average Meters
    eval_names = ['fsd', 'scl2', 'dtw', 'lpips', 'wd', 'color_diff_l1', 'color_diff_l2']
    our_metrics = AverageMetersDict(names=eval_names)
    baseline_metrics = AverageMetersDict(names=eval_names)
    snp_metrics = AverageMetersDict(names=eval_names)
    snp_plus_metrics = AverageMetersDict(names=eval_names)
    dataset_metrics = AverageMetersDict(names=['color_diff_l1', 'color_diff_l2'])

    # ======= Run ========================
    for iter in range(args.n_iters_dataloader):
        print(f'Iter : {iter} / {args.n_iters_dataloader}')
        for idx, batch in enumerate(test_loader):
            print(f'{idx} / {len(test_loader)}')
            data = dict_to_device(batch, to_skip=['strokes', 'time_steps'])
            # ======= Dataset Metrics ====
            ref_l1, ref_l2 = compute_color_difference(data['strokes_seq'])
            dataset_metrics.update(dict(color_diff_l1=ref_l1.item(), color_diff_l2=ref_l2.item()),
                                   data['strokes_seq'].shape[0])
            # ======= Predict   ===========
            # Our

            our_predictions, our_vis = eval_model(data=data, net=model, metric_logger=our_metrics,
                                                  is_our=True)
            if args.use_pt:
                pt_predictions, pt_vis = eval_model(data=data, net=baseline, metric_logger=baseline_metrics)
            if args.use_snp:
                snp_predictions, snp_vis = eval_model(data=data, net=renderer, metric_logger=snp_metrics)
            if args.use_snp2:
                snp2_predictions, snp2_vis = eval_model(data=data, net=snp_plus, metric_logger=snp_plus_metrics)

            # Add to the list
            our.append(our_predictions)
            visual_our.append(our_vis)
            if args.use_pt:
                pt.append(pt_predictions)
                visual_pt.append(pt_vis)
            if args.use_snp:
                snp.append(snp_predictions)
                visual_snp.append(snp_vis)
            if args.use_snp2:
                snp2.append(snp2_predictions)
                visual_snp2.append(snp2_vis)
            # Original
            original.append(batch['strokes_seq'])
            ctx.append(batch['strokes_ctx'])
            orig_vis = etools.render_frames(batch['strokes_seq'], data, renderer)
            ctx_vis = etools.render_frames(batch['strokes_ctx'], data, renderer)
            visual_original.append(orig_vis['frames'])
            visual_ctx.append(ctx_vis['frames'])

    # Aggregate metrics
    results = {}
    visual_original = np.concatenate(visual_original)
    visual_ctx = np.concatenate(visual_ctx)
    results.update(dataset_metrics.get_avg(header='dataset_'))

    results.update(our_metrics.get_avg(header='our_'))
    our_fid = fsdMetric(generated=our, original=original, ctx=ctx)
    visual_our = np.concatenate(visual_our)
    results.update({f'our_{k}': v for k, v in our_fid.items()})

    if args.use_pt:
        results.update(baseline_metrics.get_avg(header='pt_'))
        pt_fid = fsdMetric(generated=pt, original=original, ctx=ctx)
        visual_pt = np.concatenate(visual_pt)
        results.update({f'pt_{k}': v for k, v in pt_fid.items()})

    if args.use_snp:
        results.update(snp_metrics.get_avg(header='snp_'))
        snp_fid = fsdMetric(generated=snp, original=original, ctx=ctx)
        visual_snp = np.concatenate(visual_snp)
        results.update({f'snp_{k}': v for k, v in snp_fid.items()})

    if args.use_snp2:
        results.update(snp_plus_metrics.get_avg(header='snp++_'))
        snp2_fid = fsdMetric(generated=snp2, original=original, ctx=ctx)
        visual_snp2 = np.concatenate(visual_snp2)
        results.update({f'snp++_{k}': v for k, v in snp2_fid.items()})

    # Frechet Video Distance
    if args.fvd:
        fvd = FVD()
        fvd_our = fvd(reference_observations=np.concatenate((visual_ctx, visual_original), axis=1),
                      generated_observations=np.concatenate((visual_ctx, visual_our), axis=1))
        results.update({'our_fvd': fvd_our})
        if args.use_pt:
            fvd_pt = fvd(reference_observations=np.concatenate((visual_ctx, visual_original), axis=1),
                         generated_observations=np.concatenate((visual_ctx, visual_pt), axis=1))
            results.update({'pt_fvd': fvd_pt})
        if args.use_snp:
            fvd_snp = fvd(reference_observations=np.concatenate((visual_ctx, visual_original), axis=1),
                          generated_observations=np.concatenate((visual_ctx, visual_snp), axis=1))
            results.update({'snp_fvd': fvd_snp})
        if args.use_snp2:
            fvd_snp2 = fvd(reference_observations=np.concatenate((visual_ctx, visual_original), axis=1),
                           generated_observations=np.concatenate((visual_ctx, visual_snp2), axis=1))
            results.update({'snp++_fvd': fvd_snp2})

    # save
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, f'results.pkl'), 'wb') as f:
        pkl.dump(results, f)
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.to_csv(os.path.join(output_path, 'metrics.csv'))

if __name__ == '__main__':
    # Extra parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=False)
    parser.add_argument("--config", default='/home/eperuzzo/brushstrokes-generation/configs/eval/eval_v2.yaml')

    parser.add_argument("--output_path", type=str, default='/home/eperuzzo/eval_metrics/')
    parser.add_argument("--pt_checkpoint", type=str,
                        default='/home/eperuzzo/PaintTransformerPaddle/inference/paint_best.pdparams')
    parser.add_argument('--fvd', action='store_true')
    parser.add_argument("--n_samples", type=int, default=1,
                        help="number of samples to test lpips, diverstiy in generation")
    parser.add_argument("-n", "--n_iters_dataloader", default=1, type=int)
    parser.add_argument("--use-pt", action='store_true', default=False)
    parser.add_argument("--use-snp", action='store_true', default=False)
    parser.add_argument("--use-snp2", action='store_true', default=False)
    args = parser.parse_args()

    # Run
    main(args)
