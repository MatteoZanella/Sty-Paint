import argparse
import os
import glob
import pickle as pkl
import random

from model.utils.utils import dict_to_device, AverageMetersDict
from model.utils.parse_config import ConfigParser
from model import build_model
from model.dataset import StrokesDataset
from torch.utils.data import DataLoader
import torch
import paddle

from dataset_acquisition.decomposition.painter import Painter
from dataset_acquisition.decomposition.utils import load_painter_config

from evaluation.metrics import FeaturesDiversity, LPIPSDiversityMetric, WassersteinDistance, compute_fid
from evaluation.metrics import maskedL2, compute_dtw, compute_color_difference
from evaluation.fvd import FVD
from evaluation.tools import compute_features

from evaluation.paint_transformer.model import PaintTransformer as PaddlePT
import pandas as pd
import evaluation.tools as etools
import numpy as np
import copy
from einops import rearrange, repeat
import warnings
warnings.filterwarnings("ignore")


def np_and_save(x, path, name):
    if not x:
        return
    x = np.concatenate(x)
    np.save(os.path.join(path, name), x)

def get_features(x, ctx=None):
    if isinstance(x, list):
        x = np.concatenate(x, axis=0)
    feat_x = compute_features(x)
    if ctx:
        if isinstance(ctx, list):
            ctx = np.concatenate(ctx)
        feat_x_ctx = compute_features(np.concatenate((ctx, x), axis=1))
        return feat_x, feat_x_ctx
    else:
        return feat_x

def eval_fid(x, reference, reference_ctx, ctx):
    x_feat, x_ctx_feat = get_features(x, ctx=ctx)
    fid = compute_fid(reference, x_feat, compute_mean_cov=True)
    ctx_fid = compute_fid(reference_ctx, x_ctx_feat, compute_mean_cov=True)
    return fid, ctx_fid

def eval_model(data, net, metric_logger, is_our, renderer):

    Wdist = WassersteinDistance()
    if args.lpips and is_our:
        LPIPS = LPIPSDiversityMetric()
        feature_div = FeaturesDiversity()

    ctx = data['strokes_ctx'].cpu()
    targets = data['strokes_seq'].cpu()
    bs = targets.size(0)

    if is_our:
        net.eval()
        with torch.no_grad() :
            predictions = net.generate(data, n_samples=args.n_samples)["fake_data_random"]
    else:
        predictions = net.generate(data)

    predictions = etools.check_strokes(predictions)   # clamp in range [0,1]

    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()

    visuals = etools.render_frames(predictions, data, renderer)

    if args.lpips and is_our:
        predictions = rearrange(predictions, '(bs n_samples) L n_params -> bs n_samples L n_params', n_samples=args.n_samples)
        visuals['frames'] = rearrange(visuals['frames'], '(bs n_samples) L H W ch -> bs n_samples L H W ch', n_samples=args.n_samples)
        visuals['alphas'] = rearrange(visuals['alphas'], '(bs n_samples) L H W ch -> bs n_samples L H W ch', n_samples=args.n_samples)

        lpips_val = LPIPS(visuals['frames'], visuals['alphas'])
        metric_logger.update(dict(lpips = lpips_val.item()), bs)


        visuals['frames'] = visuals['frames'][:, 0]
        visuals['alphas'] = visuals['alphas'][:, 0]
        predictions = predictions[:, 0]

    # color difference
    color_diff_l1, color_diff_l2 = compute_color_difference(predictions)
    maskedl2 = maskedL2(data['img'], visuals['frames'], visuals['alphas'])
    wd = Wdist(targets[:, :, :5], torch.tensor(predictions[:, :, :5]))
    dtw = compute_dtw(targets, predictions)

    # record metrics
    metric_logger.update(
        dict(
            maskedL2 = maskedl2.item(),
            wd = wd.item(),
            color_diff_l1 = color_diff_l1.item(),
            color_diff_l2 = color_diff_l2.item(),
            dtw = dtw.item()), bs)

    return predictions, visuals['frames']


def main(args, exp_name):

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
    test_loader = DataLoader(dataset=dataset_test, batch_size=64, shuffle=False, pin_memory=False)
    print(f'Test : {len(dataset_test)} samples')

    # ======= Create Models ========================
    # Renderer (Stylized Neural Painting)
    render_config = load_painter_config(config["renderer"]["painter_config"])
    renderer = Painter(args=render_config)
    if args.use_snp:
        snp_plus_config = copy.deepcopy(render_config)
        snp_plus_config.with_kl_loss = True
        snp_plus = Painter(args=snp_plus_config)


    print(f'Processing: {exp_name}')
    tmp_path = os.path.join(args.checkpoint_base, exp_name)
    if os.path.exists(os.path.join(tmp_path, 'latest.pth.tar')):
        checkpoint_path = os.path.join(tmp_path, 'latest.pth.tar')
    else:
        checkpoint_path = sorted(glob.glob(os.path.join(tmp_path, '*.pth.tar')))[-1]
    if len(checkpoint_path) == 0:
        print(f'No checkpoint found at : {tmp_path}, skipping to the next!')

    # Output Path
    output_path = os.path.join(args.output_path, exp_name)

    # load checkpoint, update model config based on the stored config
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    config.update(dict(model=ckpt["config"]["model"]))

    model = build_model(config)
    msg = model.load_state_dict(ckpt["model"], strict=False)
    print(f'==> Loading model form {checkpoint_path}, with : {msg}')
    model.cuda()
    model.eval()

    # basline Paint Transformer
    baseline = PaddlePT(model_path=args.checkpoint_baseline, config=render_config)

    n_files = len(dataset_test) * args.n_iters_dataloader
    print('Number of files')
    # ======= Metrics ========================
    original, ctx, our, pt, snp, snp2 = [], [], [], [], [], []
    visual_original, visual_our, visual_pt, visual_snp, visual_snp2 = [], [], [], [], []

    # Average Meters
    eval_names = ['wd', 'maskedL2', 'color_diff_l1', 'color_diff_l2', 'dtw', 'lpips']
    our_metrics = AverageMetersDict(names=eval_names)
    baseline_metrics = AverageMetersDict(names=eval_names)
    snp_metrics = AverageMetersDict(names=eval_names)
    snp_plus_metrics = AverageMetersDict(names = eval_names)
    dataset_metrics = AverageMetersDict(names=['color_diff_l1', 'color_diff_l2'])

    # ======= Run ========================
    for iter in range(args.n_iters_dataloader) :
        print(f'Iter : {iter} / {args.n_iters_dataloader}')
        for idx, batch in enumerate(test_loader) :
            print(f'{idx} / {len(test_loader)}')
            data = dict_to_device(batch, to_skip=['strokes', 'time_steps'])
            # ======= Baseline Metrics ====
            ref_l1, ref_l2 = compute_color_difference(data['strokes_seq'])
            dataset_metrics.update(dict(color_diff_l1=ref_l1.item(), color_diff_l2 = ref_l2.item()), data['strokes_seq'].shape[0])
            # ======= Predict   ===========
            our_predictions, our_vis = eval_model(data=data, net=model, metric_logger=our_metrics, is_our=True, renderer=renderer)
            if args.use_pt:
                pt_predictions, pt_vis =eval_model(data=data, net=baseline, metric_logger=baseline_metrics, is_our=False, renderer=renderer)
            if args.use_snp:
                snp_predictions, snp_vis =eval_model(data=data, net=renderer, metric_logger=snp_metrics, is_our=False, renderer=renderer)

                snp2_predictions, snp2_vis = eval_model(data=data, net=snp_plus, metric_logger=snp_plus_metrics, is_our=False, renderer=renderer)

            # Add to the list
            our.append(our_predictions)
            visual_our.append(our_vis)
            if args.use_pt:
                pt.append(pt_predictions)
                visual_pt.append(pt_vis)
            if args.use_snp:
                snp.append(snp_predictions)
                visual_snp.append(snp_vis)
                snp2.append(snp2_predictions)
                visual_snp2.append(snp2_vis)
            # Original
            original.append(batch['strokes_seq'])
            ctx.append(batch['strokes_ctx'])
            orig_vis = etools.render_frames(batch['strokes_seq'], data, renderer)
            visual_original.append(orig_vis['frames'])

    # Aggragte metrics
    results = {}
    original_feat, original_ctx_feat = get_features(original, ctx=ctx)
    visual_original = np.concatenate(visual_original)
    results.update(dataset_metrics.get_avg(header='dataset_'))


    results.update(our_metrics.get_avg(header='our_'))
    our_fid, our_ctx_fid = eval_fid(x=our, reference=original_feat, reference_ctx=original_ctx_feat, ctx=ctx)
    visual_our = np.concatenate(visual_our)
    results.update({f'our_{k}' : v for k, v in our_fid.items()})
    results.update({f'our_ctx_{k}' : v for k, v in our_ctx_fid.items()})

    if args.use_pt:
        results.update(baseline_metrics.get_avg(header='pt_'))
        pt_fid, pt_ctx_fid = eval_fid(x=pt, reference=original_feat, reference_ctx=original_ctx_feat, ctx=ctx)
        visual_pt = np.concatenate(visual_pt)
        results.update({f'pt_{k}' : v for k, v in pt_fid.items()})
        results.update({f'pt_ctx_{k}' : v for k, v in pt_ctx_fid.items()})

    if args.use_snp:
        results.update(snp_metrics.get_avg(header='snp_'))
        snp_fid, snp_ctx_fid = eval_fid(x=snp, reference=original_feat, reference_ctx=original_ctx_feat, ctx=ctx)
        results.update({f'snp_{k}' : v for k, v in snp_fid.items()})
        results.update({f'snp_ctx_{k}' : v for k, v in snp_ctx_fid.items()})

        results.update(snp_plus_metrics.get_avg(header='snp++_'))
        snp2_fid, snp2_ctx_fid = eval_fid(x=snp2, reference=original_feat, reference_ctx=original_ctx_feat, ctx=ctx)
        results.update({f'snp++_{k}' : v for k, v in snp2_fid.items()})
        results.update({f'snp++_ctx_{k}' : v for k, v in snp2_ctx_fid.items()})
    # FVD
    if args.fvd:
        fvd = FVD()

        fvd_our = fvd(reference_observations=visual_original, generated_observations=visual_our)
        results.update({'our_fvd' : fvd_our})
        if args.use_pt:
            fvd_pt = fvd(reference_observations=visual_original, generated_observations=visual_pt)
            results.update({'pt_fvd' : fvd_pt})
        if args.use_snp:
            fvd_snp = fvd(reference_observations=visual_original, generated_observations=visual_snp)
            fvd_snp2 = fvd(reference_observations=visual_original, generated_observations=visual_snp2)
            results.update({'snp_fvd' : fvd_snp})
            results.update({'snp++_fvd' : fvd_snp2})

    # save
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, f'results.pkl'), 'wb') as f :
        pkl.dump(results, f)
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.to_csv(os.path.join(output_path, 'metrics.csv'))

    # Save to output
    np_and_save(original, path=output_path, name='original')
    np_and_save(ctx, path=output_path, name='ctx')
    np_and_save(our, path=output_path, name='our')
    np_and_save(pt, path=output_path, name='pt')
    np_and_save(snp, path=output_path, name='snp')
    np_and_save(snp2, path=output_path, name='snp2')


if __name__ == '__main__' :
    # Extra parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_base", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=False)
    parser.add_argument("--config", default='/home/eperuzzo/brushstrokes-generation/configs/eval/eval_v2.yaml')

    parser.add_argument("--output_path", type=str, default='/home/eperuzzo/eval_metrics/')
    parser.add_argument("--checkpoint_baseline", type=str,
                        default='/home/eperuzzo/PaintTransformerPaddle/inference/paint_best.pdparams')
    parser.add_argument('--lpips', action='store_true')
    parser.add_argument('--fvd', action='store_true')
    parser.add_argument("--n_samples", type=int, default=1,
                        help="number of samples to test lpips, diverstiy in generation")
    parser.add_argument("-n", "--n_iters_dataloader", default=1, type=int)
    parser.add_argument("--use-pt", action='store_true', default=False)
    parser.add_argument("--use-snp", action='store_true',default=False)
    args = parser.parse_args()

    if args.lpips and args.n_samples == 1:
        print('To evalutate LPIPS need more than 1 sample')
        args.n_samples = 3

    # List
    experiments = os.listdir(args.checkpoint_base)
    print(f'Experiments to evaluate : {experiments}')
    for exp in experiments:
        main(args, exp)