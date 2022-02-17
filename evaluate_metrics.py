import argparse
import os
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

from evaluation.metrics import FeaturesDiversity, LPIPSDiversityMetric, WassersteinDistance, FDMetricIncremental, FDWithContextMetricIncremental
from evaluation.metrics import maskedL2, compute_dtw, compute_color_difference
from evaluation.fvd import FVD

from evaluation.paint_transformer.model import PaintTransformer as PaddlePT
# from evaluation.paint_transformer.torch_implementation import PaintTransformer
import pandas as pd
import evaluation.tools as etools
import numpy as np
import copy
import warnings
warnings.filterwarnings("ignore")


def eval_snp(data, net, metric_snp, fd, fd_ctx, metric_snp_plus, fd_plus, fd_ctx_plus, renderer):
    Wdist = WassersteinDistance()

    ctx = data['strokes_ctx'].cpu()
    targets = data['strokes_seq'].cpu()
    bs = targets.size(0)


    snp, snp_plus = net.generate(data)
    #######################################################
    # SNP
    snp = etools.check_strokes(snp)  # clamp in range [0,1]
    snp_visuals = etools.render_frames(snp, data, renderer)
    # color difference
    color_diff_l1, color_diff_l2 = compute_color_difference(snp)
    maskedl2 = maskedL2(data['img'], snp_visuals['frames'], snp_visuals['alphas'])
    wd = Wdist(targets[:, :, :5], torch.tensor(snp[:, :, :5]))
    dtw = compute_dtw(targets, snp)
    fd.update_queue(original=targets,
                    generated=snp)
    fd_ctx.update_queue(original=np.concatenate((ctx, targets), axis=1),
                        generated=np.concatenate((ctx, snp), axis=1))

    # record metrics
    metric_snp.update(
        dict(
            maskedL2=maskedl2.item(),
            wd=wd.item(),
            color_diff_l1=color_diff_l1.item(),
            color_diff_l2=color_diff_l2.item(),
            dtw=dtw.item()), bs)


    ##################
    # SNP plus
    snp_plus = etools.check_strokes(snp_plus)  # clamp in range [0,1]
    snp_plus_visuals = etools.render_frames(snp_plus, data, renderer)
    # color difference
    color_diff_l1, color_diff_l2 = compute_color_difference(snp_plus)
    maskedl2 = maskedL2(data['img'], snp_plus_visuals['frames'], snp_plus_visuals['alphas'])
    wd = Wdist(targets[:, :, :5], torch.tensor(snp_plus[:, :, :5]))
    dtw = compute_dtw(targets, snp_plus)
    fd_plus.update_queue(original=targets,
                    generated=snp_plus)
    fd_ctx_plus.update_queue(original=np.concatenate((ctx, targets), axis=1),
                        generated=np.concatenate((ctx, snp_plus), axis=1))

    # record metrics
    metric_snp_plus.update(
        dict(
            maskedL2=maskedl2.item(),
            wd=wd.item(),
            color_diff_l1=color_diff_l1.item(),
            color_diff_l2=color_diff_l2.item(),
            dtw=dtw.item()), bs)

def eval_model(data, net, metric_logger, fd, fd_ctx, is_our, renderer):

    Wdist = WassersteinDistance()
    # LPIPS = LPIPSDiversityMetric()
    feature_div = FeaturesDiversity()

    ctx = data['strokes_ctx'].cpu()
    targets = data['strokes_seq'].cpu()
    bs = targets.size(0)

    # if is_our:
    #     net.eval()
    #     print(f'Model is training: {net.training}')
    if is_our:
        net.eval()
        with torch.no_grad() :
            predictions = net.generate(data, n_samples=1)["fake_data_random"]
    else:
        predictions = net.generate(data)

    predictions = etools.check_strokes(predictions)   # clamp in range [0,1]

    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()

    visuals = etools.render_frames(predictions, data, renderer)

    # color difference
    color_diff_l1, color_diff_l2 = compute_color_difference(predictions)
    maskedl2 = maskedL2(data['img'], visuals['frames'], visuals['alphas'])
    wd = Wdist(targets[:, :, :5], torch.tensor(predictions[:, :, :5]))
    dtw = compute_dtw(targets, predictions)
    fd.update_queue(original=targets,
                    generated=predictions)
    fd_ctx.update_queue(original=np.concatenate((ctx, targets), axis=1),
                        generated=np.concatenate((ctx, predictions), axis=1))

    # record metrics
    metric_logger.update(
        dict(
            maskedL2 = maskedl2.item(),
            wd = wd.item(),
            color_diff_l1 = color_diff_l1.item(),
            color_diff_l2 = color_diff_l2.item(),
            dtw = dtw.item()), bs)


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
        snp_plus_config.beta_kl = 0.00000001
        snp_plus_config.with_kl_loss = True
        snp_plus = Painter(args=snp_plus_config)


    print(f'Processing: {exp_name}')
    checkpoint_path = os.path.join(args.checkpoint_base, exp_name, 'latest.pth.tar')

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
    # fvd = FVD()
    fd_our = FDMetricIncremental()
    fd_our_w_context = FDWithContextMetricIncremental(name='our', seq_len=18, K=10)

    fd_baseline = FDMetricIncremental()
    fd_baseline_w_context = FDWithContextMetricIncremental(name='pt', seq_len=18, K=10)

    fd_snp = FDMetricIncremental()
    fd_snp_w_context = FDWithContextMetricIncremental(name='snp', seq_len=18, K=10)

    fd_snp_plus = FDMetricIncremental()
    fd_snp_plus_w_context = FDWithContextMetricIncremental(name='snp_plus', seq_len=18, K=10)

    # Average Meters
    eval_names = ['wd', 'maskedL2', 'color_diff_l1', 'color_diff_l2', 'dtw']
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
            eval_model(data=data, net=model, metric_logger=our_metrics, fd=fd_our, fd_ctx=fd_our_w_context,
                       is_our=True, renderer=renderer)
            if args.use_pt:
                eval_model(data=data, net=baseline, metric_logger=baseline_metrics, fd=fd_baseline, fd_ctx=fd_baseline_w_context,
                           is_our=False, renderer=renderer)
            if args.use_snp:
                eval_model(data=data, net=renderer, metric_logger=snp_metrics, fd=fd_snp,
                           fd_ctx=fd_snp_w_context,
                           is_our=False,
                           renderer=renderer)

                eval_model(data=data, net=snp_plus, metric_logger=snp_plus_metrics,
                           fd=fd_snp_plus,
                           fd_ctx=fd_snp_plus_w_context,
                           is_our=False,
                           renderer=renderer)


                '''
                eval_snp(data=data, net=renderer,
                         metric_snp=snp_metrics,
                         fd=fd_snp,
                         fd_ctx=fd_snp_w_context,
                         metric_snp_plus=snp_plus_metrics,
                         fd_plus=fd_snp_plus,
                         fd_ctx_plus=fd_snp_plus_w_context,
                         renderer=renderer)
                '''

    #  TODO:
    results = {}
    results.update(our_metrics.get_avg(header='our_'))
    results.update({f'our_fd_{k}' : v for k, v in fd_our.compute_fd().items()})
    results.update({f'our_fd_wctx_{k}' : v for k, v in fd_our_w_context.compute_fd().items()})

    if args.use_pt:
        results.update(baseline_metrics.get_avg(header='baseline_'))
        results.update({f'baseline_fd_{k}' : v for k, v in fd_baseline.compute_fd().items()})
        results.update({f'baseline_fd_wctx_{k}' : v for k, v in fd_baseline_w_context.compute_fd().items()})

    if args.use_snp:
        results.update(snp_metrics.get_avg(header='snp_'))
        results.update({f'snp_fd_{k}' : v for k, v in fd_snp.compute_fd().items()})
        results.update({f'snp_fd_wctx_{k}' : v for k, v in fd_snp_w_context.compute_fd().items()})

        results.update(snp_plus_metrics.get_avg(header='snp++_'))
        results.update({f'snp++_fd_{k}' : v for k, v in fd_snp_plus.compute_fd().items()})
        results.update({f'snp++_fd_wctx_{k}' : v for k, v in fd_snp_plus_w_context.compute_fd().items()})

    results.update(dataset_metrics.get_avg(header='dataset_'))

    # save
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, f'results.pkl'), 'wb') as f :
        pkl.dump(results, f)
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.to_csv(os.path.join(output_path, 'metrics.csv'))


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
    parser.add_argument("--n_samples_lpips", type=int, default=3,
                        help="number of samples to test lpips, diverstiy in generation")
    parser.add_argument("-n", "--n_iters_dataloader", default=1, type=int)
    parser.add_argument("--use-pt", action='store_true', default=False)
    parser.add_argument("--use-snp", action='store_true',default=False)
    args = parser.parse_args()

    # List
    experiments = os.listdir(args.checkpoint_base)
    print(f'Experiments to evaluate : {experiments}')
    for exp in experiments:
        main(args, exp)