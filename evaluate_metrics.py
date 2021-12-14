import argparse
import os
import pickle as pkl

from model.utils.utils import dict_to_device, AverageMeter
from model.utils.parse_config import ConfigParser
from model import build_model
from model.dataset import StrokesDataset
from torch.utils.data import DataLoader

from dataset_acquisition.decomposition.painter import Painter
from dataset_acquisition.decomposition.utils import load_painter_config
import torch

from evaluation.metrics import FeaturesDiversity, LPIPSDiversityMetric, WassersteinDistance, FDMetricIncremental
from evaluation.metrics import maskedL2, compute_dtw
from evaluation.fvd import FVD

from evaluation.paint_transformer.model import PaintTransformer as PaddlePT
#from evaluation.paint_transformer.torch_implementation import PaintTransformer
import pandas as pd
import evaluation.tools as etools
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def count_parameters(net) :
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def to_df(data):
    out = dict()
    for name, results in data.items():
        out[name] = dict()
        for metric in results.keys() :
            if metric == 'fd' :
                out[name]['fd_all'] = results['fd'][1]['all']
                out[name]['fd_position'] = results['fd'][1]['position']
                out[name]['fd_color'] = results['fd'][1]['color']
            else :
                out[name][metric] = results[metric].avg

    return out

def compute_color_difference(x):
    x = x[:, :, 5:]
    if torch.is_tensor(x):
        l1 = torch.abs(torch.diff(x, dim=1)).mean()
        l2 = torch.pow(torch.diff(x, dim=1), 2).mean()
    else:
        l1 = np.abs(np.diff(x, axis=1)).sum(axis=1).mean()
        l2 = np.square(np.diff(x, axis=1)).mean()

    return l1, l2

if __name__ == '__main__' :
    # Extra parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, required=True)
    #parser.add_argument("--checkpoint2", type=str, required=True)
    parser.add_argument("--config", default='/home/eperuzzo/brushstrokes-generation/configs/train/conf_sibiu.yaml')

    parser.add_argument("--output_path", type=str, default='/home/eperuzzo/eval_metrics/')
    parser.add_argument("--checkpoint_baseline", type=str,
                        default='/home/eperuzzo/PaintTransformer/inference/paint_best.pdparams')
    parser.add_argument('--lpips', action='store_true')
    parser.add_argument('--fvd', action='store_true')
    parser.add_argument("--n_samples_lpips", type=int, default=3,
                        help="number of samples to test lpips, diverstiy in generation")
    parser.add_argument("--n_iters_dataloader", default=1, type=int)
    args = parser.parse_args()


    # Create config
    c_parser = ConfigParser(args.config, isTrain=False)
    c_parser.parse_config(args)
    config = c_parser.get_config()
    print(config)

    # Seed
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    # Test
    dataset_test = StrokesDataset(config, isTrain=False)
    test_loader = DataLoader(dataset=dataset_test, batch_size=16, shuffle=True, pin_memory=False)
    print(f'Test : {len(dataset_test)} samples')

    # ======= Create Models ========================
    # Renderer (Stylized Neural Painting)
    render_config = load_painter_config(config["render"]["painter_config"])
    renderer = Painter(args=render_config)

    model = build_model(config)
    print(f'==> Loading model form {args.model_checkpoint}')
    model.load_state_dict(torch.load(args.model_checkpoint)["model"])
    model.cuda()
    model.eval()

    models = dict(
        our = model.eval(),
        #our_plus = net2.eval(),
        paint_transformer = PaddlePT(model_path=args.checkpoint_baseline, config=render_config))

    n_files = len(dataset_test) * args.n_iters_dataloader
    print('Number of files')
    # ======= Metrics ========================
    LPIPS = LPIPSDiversityMetric()
    Wdist = WassersteinDistance()
    feature_div = FeaturesDiversity()
    fvd = FVD()

    # Average Meters
    average_meters = dict()

    for key in models.keys():
        average_meters.update({key : dict(
                           fd = FDMetricIncremental(),
                           dtw = AverageMeter(),
                           masked_l2=AverageMeter(),
                           fvd = AverageMeter(),
                           lpips=AverageMeter(),
                           feature_div=AverageMeter(),
                           color_l1=AverageMeter(),
                           color_l2 = AverageMeter(),
                           wd=AverageMeter())})

    average_meters.update({'original' : dict(
        masked_l2 = AverageMeter(),
        color_l1 = AverageMeter(),
        color_l2 = AverageMeter(),
    )})
    # average_meters.update({'model_sc' : dict(
    #     fd = FDMetricIncremental(),
    #     dtw=AverageMeter(name='dtw'),
    #     masked_l2=AverageMeter(name='Masked L2 / Area'),
    #     lpips=AverageMeter(name='lpips'),
    #     feature_div=AverageMeter(name='features_div'),
    #     color_difference=AverageMeter(name='avg'),
    #     wd=AverageMeter(name='wd'))})

    # ======= Run ========================
    for iter in range(args.n_iters_dataloader):
        print(f'Iter : {iter} / {args.n_iters_dataloader}')
        for idx, batch in enumerate(test_loader):
            print(f'{idx} / {len(test_loader)}')
            data = dict_to_device(batch, to_skip=['strokes', 'time_steps'])
            targets = data['strokes_seq']
            starting_point = batch['canvas'][0].permute(1,2,0).cpu().numpy()
            bs = targets.size(0)
            # ======= Predict   ===========
            predictions = dict()
            visuals = dict()
            for name, model in models.items():
                preds = model.generate(data)
                if name == 'our' or name == 'our_plus':
                    preds = etools.check_strokes(preds)
                if torch.is_tensor(preds):
                    preds = preds.cpu().numpy()
                predictions.update({name : preds})
            #predictions.update({'model_sc' : etools.sample_color(predictions['our'], batch['img'])})

            for name, params in predictions.items() :
                visuals.update({name : etools.render_frames(params, batch, renderer)})
            visuals.update({'original' : etools.render_frames(batch['strokes_seq'], batch, renderer)})

            # ========================================
            # Compute cumulative difference
            for name, preds in predictions.items():
                color_diff_l1, color_diff_l2 = compute_color_difference(preds)
                average_meters[name]['color_l1'].update(color_diff_l1.item(), bs)
                average_meters[name]['color_l2'].update(color_diff_l2.item(), bs)

            color_diff_l1, color_diff_l2 = compute_color_difference(batch['strokes_seq'])  # original strokes
            average_meters['original']['color_l1'].update(color_diff_l1.item(), bs)
            average_meters['original']['color_l2'].update(color_diff_l2.item(), bs)
            # =======================================
            # FVD
            if args.fvd:
                try:
                    for name, model in visuals.items():
                        if name == 'original':
                            continue
                        res = fvd(reference_observations=visuals['original']['frames'], generated_observations=visuals[name]['frames'])
                        average_meters[name]['fvd'].update(res, bs)
                except:
                    print('Skipping this iteration')
            # =======================================
            # Maksed l2
            for name, model in visuals.items():
                tmp = maskedL2(batch['img'], visuals[name]['frames'], visuals[name]['alphas'])
                average_meters[name]['masked_l2'].update(tmp.item(), bs)

            # =========================================
            # Wasserstein/Frechet/Dtw Distance
            for name in predictions.keys():
                wd = Wdist(batch['strokes_seq'][:, :, :5], torch.tensor(predictions[name][:, :, :5]))
                average_meters[name]['fd'].update_queue(batch['strokes_seq'], predictions[name])  # keep all
                dtw = compute_dtw(batch['strokes_seq'], predictions[name])

                average_meters[name]['wd'].update(wd.item(), bs)
                #average_meters[name]['fd'].update(fd.item(), bs)
                average_meters[name]['dtw'].update(dtw.item(), bs)

            #####
            # Diversity
            predictions_lpips = dict()

            for key, model in models.items():
                if key == 'paint_transformer' or key == 'model_sc':
                    continue
                predictions_lpips[key] = dict()
                for n in range(args.n_samples_lpips):
                    predictions_lpips[key][n] = model.generate(data)

            if args.lpips and idx % 5 == 0 :
                visuals_lpips = etools.render_lpips(inp=predictions_lpips, renderer=renderer,
                                             batch=batch, bs=bs, n_samples=args.n_samples_lpips)

                for name in visuals_lpips.keys():
                    lpips_score = LPIPS(visuals_lpips[name])
                    average_meters[name]['lpips'].update(lpips_score.item(), bs)

            preds = etools.prepare_feature_difference(predictions_lpips, bs, args.n_samples_lpips)
            for name in preds.keys():
                feat_div_score = feature_div(preds[name])
                average_meters[name]['feature_div'].update(feat_div_score.item(), bs)

    for key in average_meters.keys():
        if 'fd' in average_meters[key]:
            average_meters[key]['fd'] = average_meters[key]['fd'].compute_fd()

    os.makedirs(args.output_path, exist_ok=True)
    results = to_df(average_meters)

    with open(os.path.join(args.output_path, f'results.pkl'), 'wb') as f:
        pkl.dump(results, f)
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.to_csv(os.path.join(args.output_path, 'metrics.csv'))