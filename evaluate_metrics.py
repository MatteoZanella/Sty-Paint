import argparse
import os
import pickle as pkl

from model.utils.utils import dict_to_device, AverageMeter
from model.utils.parse_config import ConfigParser
from model import model, model_2_steps
#from model.paint_transformer.torch_implementation import PaintTransformer
from model.dataset import StrokesDataset
from torch.utils.data import DataLoader

from dataset_acquisition.decomposition.painter import Painter
from dataset_acquisition.decomposition.utils import load_painter_config
import torch
import numpy as np

from evaluation.metrics import FDMetric, FeaturesDiversity, LPIPSDiversityMetric, WassersteinDistance, FDMetricIncremental
from evaluation.metrics import maskedL2, compute_dtw

from evaluation.paint_transformer.model import PaintTransformer as PaddlePT
import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F

def count_parameters(net) :
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def sample_color(params, imgs) :
    b, _, h, w = imgs
    n_strokes = 8
    img_temp = imgs.unsqueeze(1).repeat(1, n_strokes, 1, 1, 1).view(b * n_strokes, 3, h, w)
    grid = params[:, :, :2].view(b * n_strokes, 1, 1, 2).contiguous()
    color = F.grid_sample(img_temp, 2 * grid - 1, align_corners=False).view(b, n_strokes, 3).contiguous()
    color = color.repeat(1, 1, 2)
    out_params = torch.cat((params.clone()[:, :, :5], color), dim=-1)
    return out_params


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


def build_models(config):
    model_type = config["model"]["model_type"]
    if model_type == "full":
        net = model.InteractivePainter(config)
    elif model_type == "2_steps":
        net = model_2_steps.InteractivePainter(config)
    else:
        raise NotImplementedError()
    net.load_state_dict(torch.load(config["model"]["checkpoint"], map_location=device)["model"])
    net.to(config["device"])
    net.eval()



if __name__ == '__main__' :
    # Extra parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_1_config", type=str, required=True)
    parser.add_argument("--model_2_config", type=str, required=True)
    parser.add_argument("--config", default='../configs/train/sibiu_config.yaml')

    parser.add_argument("--output_path", type=str, default='/home/eperuzzo/eval_metrics/')
    parser.add_argument("--checkpoint_baseline", type=str,
                        default='/home/eperuzzo/OfficialPaintTransformer/inference/model.pth')
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
    dataset_test = StrokesDataset(config, isTrain=False)
    test_loader = DataLoader(dataset=dataset_test, batch_size=16, shuffle=True, pin_memory=False)
    print(f'Test : {len(dataset_test)} samples')

    # ======= Create Models ========================
    # Renderer (Stylized Neural Painting)
    render_config = load_painter_config(config["render"]["painter_config"])
    renderer = Painter(args=render_config)

    net1 = model.InteractivePainter(config)
    print(f'==> Loading model form {args.model_1_config}')
    net1.load_state_dict(torch.load(args.model_1_config, map_location=device)["model"])
    net1.to(config["device"])
    net1.eval()

    net2 = model_2_steps.InteractivePainter(config)
    print(f'==> Loading model form {args.model_2_config}')
    net2.load_state_dict(torch.load(args.model_2_config, map_location=device)["model"])
    net2.to(config["device"])
    net2.eval()

    models = dict(
        model = net1,
        model_two_steps = net2,
        baseline = PaddlePT(model_path=args.checkpoint_baseline, config=render_config))

    n_files = len(dataset_test) * args.n_iters_dataloader
    print('Number of files')
    # ======= Metrics ========================
    LPIPS = LPIPSDiversityMetric()
    Wdist = WassersteinDistance()
    Fdist = FDMetric()
    feature_div = FeaturesDiversity()

    # Average Meters
    average_meters = dict()

    for key in models.keys():
        average_meters.update({key : dict(
                           fd = FDMetricIncremental(n_files=n_files),
                           dtw = AverageMeter(name='dtw'),
                           masked_l2=AverageMeter(name='Masked L2 / Area'),
                           lpips=AverageMeter(name='lpips'),
                            feature_div=AverageMeter(name='features_div'),
                            wd=AverageMeter(name='wd'))})
    average_meters.update({'model_sc' : dict(
        fd = FDMetricIncremental(n_files=n_files),
        dtw=AverageMeter(name='dtw'),
        masked_l2=AverageMeter(name='Masked L2 / Area'),
        lpips=AverageMeter(name='lpips'),
        feature_div=AverageMeter(name='features_div'),
        wd=AverageMeter(name='wd'))})

    # ======= Run ========================
    for iter in range(args.n_iters_dataloader):
        print(f'Iter : {iter} / {args.n_iters_dataloader}')
        for idx, batch in enumerate(test_loader):
            print(f'{idx} / {len(test_loader)}')
            data = dict_to_device(batch, device, to_skip=['strokes', 'time_steps'])
            targets = data['strokes_seq']
            starting_point = batch['canvas'][0].permute(1,2,0).cpu().numpy()
            bs = targets.size(0)
            # ======= Predict   ===========
            predictions = dict()
            visuals = dict()
            for name, model in models.items():
                preds = model.generate(data)
                if torch.is_tensor(preds):
                    preds = preds.cpu().numpy()
                predictions.update({name : preds})
            predictions.update({'model_sc' : sample_color(predictions['model'], batch['img'])})

            for name, params in predictions.items() :
                visuals.update({name : render_frames(params, batch, renderer)})

            # Maksedl2
            print('MaskedL2')
            for name, model in visuals.items():
                print(name)
                tmp = maskedL2(batch['img'], visuals[name]['frames'], visuals[name]['alphas'])
                average_meters[name]['masked_l2'].update(tmp.item(), bs)

            # Wasserstein/Frechet/Dtw Distance
            print('WD/FD/DTW')
            for name in predictions.keys():
                print(name)
                wd = Wdist(batch['strokes_seq'][:, :, :5], torch.tensor(predictions[name][:, :, :5]))
                average_meters[name]['fd'].update_queue(batch['strokes_seq'], predictions[name])  # keep all
                dtw = compute_dtw(batch['strokes_seq'], predictions[name])

                average_meters[name]['wd'].update(wd.item(), bs)
                #average_meters[name]['fd'].update(fd.item(), bs)
                average_meters[name]['dtw'].update(dtw.item(), bs)

            #####
            # Diversity

            print('\nLPIPS TIME')
            predictions_lpips = dict()

            for key, model in models.items():
                if key == 'paint_transformer' or key == 'model_sc':
                    continue
                predictions_lpips[key] = dict()
                for n in range(args.n_samples_lpips):
                    predictions_lpips[key][n] = model.generate(data)

            flag = False
            if flag:#idx % 5 == 0 :
                visuals_lpips = render_lpips(inp=predictions_lpips, renderer=renderer,
                                             batch=batch, bs=bs, n_samples=args.n_samples_lpips)

                for name in visuals_lpips.keys():
                    lpips_score = LPIPS(visuals_lpips[name])
                    average_meters[name]['lpips'].update(lpips_score.item(), bs)

            preds = prepare_feature_difference(predictions_lpips, bs, args.n_samples_lpips)
            for name in preds.keys():
                feat_div_score = feature_div(preds[name])
                print(feat_div_score.item())
                average_meters[name]['feature_div'].update(feat_div_score.item(), bs)

    for key in average_meters.keys():
        average_meters[key]['fd'] = average_meters[key]['fd'].compute_fd()

    os.makedirs(args.output_path, exist_ok=True)
    for key in average_meters.keys():
        with open(os.path.join(args.output_path, f'{key}.pkl'), 'wb') as f:
            pkl.dump(average_meters[key], f)
