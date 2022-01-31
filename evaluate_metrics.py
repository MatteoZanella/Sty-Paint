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

from evaluation.metrics import FeaturesDiversity, LPIPSDiversityMetric, WassersteinDistance, FDMetricIncremental
from evaluation.metrics import maskedL2, compute_dtw, compute_color_difference
from evaluation.fvd import FVD

from evaluation.paint_transformer.model import PaintTransformer as PaddlePT
#from evaluation.paint_transformer.torch_implementation import PaintTransformer
import pandas as pd
import evaluation.tools as etools
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def eval_model(data, net, metric_logger, fd, is_our, renderer):

    Wdist = WassersteinDistance()
    targets = data['strokes_seq'].cpu()
    bs = targets.size(0)

    # if is_our:
    #     net.eval()
    #     print(f'Model is trianing: {net.training}')
    with torch.no_grad():
        if is_our:
            predictions = net(data, sample_z=True)["fake_data_random"]
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
    fd.update_queue(targets, predictions)

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
    # LPIPS = LPIPSDiversityMetric()
    # feature_div = FeaturesDiversity()
    # fvd = FVD()
    fd_our = FDMetricIncremental()
    fd_baseline = FDMetricIncremental()

    # Average Meters
    eval_names = ['wd', 'maskedL2', 'color_diff_l1', 'color_diff_l2', 'dtw']
    our_metrics = AverageMetersDict(names=eval_names)
    baseline_metrics = AverageMetersDict(names=eval_names)

    # ======= Run ========================
    for iter in range(args.n_iters_dataloader) :
        print(f'Iter : {iter} / {args.n_iters_dataloader}')
        for idx, batch in enumerate(test_loader) :
            print(f'{idx} / {len(test_loader)}')
            data = dict_to_device(batch, to_skip=['strokes', 'time_steps'])
            # ======= Predict   ===========
            eval_model(data=data, net=model, metric_logger=our_metrics, fd=fd_our, is_our=True, renderer=renderer)
            eval_model(data=data, net=baseline, metric_logger=baseline_metrics, fd=fd_baseline, is_our=False, renderer=renderer)

    #  TODO:
    results = {}
    results.update(our_metrics.get_avg(header='our_'))
    results.update({f'our_fd_{k}' : v for k, v in fd_our.compute_fd().items()})

    results.update(baseline_metrics.get_avg(header='baseline_'))
    results.update({f'baseline_fd_{k}' : v for k, v in fd_baseline.compute_fd().items()})

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
    parser.add_argument("--config", default='/home/eperuzzo/brushstrokes-generation/configs/eval/eval.yaml')

    parser.add_argument("--output_path", type=str, default='/home/eperuzzo/eval_metrics/')
    parser.add_argument("--checkpoint_baseline", type=str,
                        default='/home/eperuzzo/PaintTransformerPaddle/inference/paint_best.pdparams')
    parser.add_argument('--lpips', action='store_true')
    parser.add_argument('--fvd', action='store_true')
    parser.add_argument("--n_samples_lpips", type=int, default=3,
                        help="number of samples to test lpips, diverstiy in generation")
    parser.add_argument("-n", "--n_iters_dataloader", default=1, type=int)
    args = parser.parse_args()


    # List
    experiments = os.listdir(args.checkpoint_base)
    print(f'Experiments to evaluate : {experiments}')
    for exp in experiments:
        main(args, exp)