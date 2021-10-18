import argparse
import logging
import os
import pickle as pkl

from model.utils.utils import dict_to_device, AverageMeter
from model.utils.parse_config import ConfigParser
from model import model, model_old
from model.baseline.model import PaintTransformer
from model.dataset import StrokesDataset, EvalDataset
from torch.utils.data import DataLoader

from dataset_acquisition.decomposition.painter import Painter
from dataset_acquisition.decomposition.utils import load_painter_config
import torch

from model.evaluation.metrics import FDMetric, LPIPSMetric, MaskedL2


def count_parameters(net) :
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def get_canvas_context(x) :
    x = x['canvas_ctx'][0, -1].permute(1, 2, 0).detach().cpu().numpy()
    x = 0.5 * (x + 1)
    return x

def old_to_new(x):
    out = {
        'strokes_ctx' : x['strokes_ctx'],
        'strokes_seq' : x['strokes_seq'],
        'canvas' : x['canvas_ctx'][0, -1],
        'img' : x['img']}

    return out


def compute_metrics(preds,
                    y,
                    renderer,
                    canvas_st,
                    img_ref,
                    meters):
    # FD
    meters['fd'].update_queue(original=y, generated=preds)

    # Visual Losses
    foreground, alpha = renderer.inference(preds, canvas_start=canvas_st)
    l2, l2_norm = l2_metric(img_ref, foreground, alpha)
    meters["masked_l2"].update(l2.item(), 1)
    meters["masked_l2_over_area"].update(l2_norm.item(), 1)

def compute_lpips(data, model, meter, n, st):
    renders = []
    alphas = []
    for j in range(n) :
        pred_tmp = model.generate(data, no_z=True)
        res, alpha = renderer.inference(pred_tmp.cpu(), canvas_start=st)
        renders.append(res)
        alphas.append(alpha)

    lpips_score = lpips_metric(renders)
    meter["lpips"].update(lpips_score.item(), 1)

if __name__ == '__main__' :
    # Extra parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_new", type=str, required=True)
    parser.add_argument("--model_old", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--old_model", action="store_true")
    parser.add_argument("--output_path", type=str, default='/home/eperuzzo/eval_metrics/')
    parser.add_argument("--checkpoint_baseline", type=str,
                        default='/home/eperuzzo/PaintTransformer/inference/paint_best.pdparams')
    parser.add_argument("--n_samples", type=int, default=3,
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
    test_loader = DataLoader(dataset=dataset_test, batch_size=1, shuffle=True, pin_memory=False)
    print(f'Test : {len(dataset_test)} samples')

    # ======= Create Models ========================
    # Renderer (Stylized Neural Painting)
    render_config = load_painter_config(config["render"]["painter_config"])
    renderer = Painter(args=render_config)
    # Interactive Painter (Our)
    old_IP = model_old.InteractivePainter(config)
    old_IP.load_state_dict(torch.load(args.model_old, map_location=device)["model"])
    old_IP.to(device)

    new_IP = model.InteractivePainter(config)
    new_IP.load_state_dict(torch.load(args.model_new, map_location=device)["model"])
    new_IP.to(device)

    # Paint Transformer (baseline)
    PT = PaintTransformer(model_path=args.checkpoint_baseline,
                          config=render_config)

    # ======= Metrics ========================

    lpips_metric = LPIPSMetric()
    l2_metric = MaskedL2()

    # Average Meters
    baseline_meters = dict(fd = FDMetric(n_files=len(dataset_test) * args.n_iters_dataloader),
                           masked_l2=AverageMeter(name='Masked L2'),
                           masked_l2_over_area=AverageMeter(name='Masked L2 / Area'),)
                           #lpips_general = AverageMeter(name='LPIPSvsImage'))

    model_new_meters = dict(fd = FDMetric(n_files=len(dataset_test) * args.n_iters_dataloader),
                        masked_l2=AverageMeter(name='Masked L2'),
                        masked_l2_over_area=AverageMeter(name='Masked L2 / Area'),
                        lpips=AverageMeter(name='LPIPS'))

    model_old_meters = dict(fd = FDMetric(n_files=len(dataset_test) * args.n_iters_dataloader),
                        masked_l2=AverageMeter(name='Masked L2'),
                        masked_l2_over_area=AverageMeter(name='Masked L2 / Area'),
                        lpips=AverageMeter(name='LPIPS'),)

    # ======= Run ========================
    for iter in range(args.n_iters_dataloader):
        print(f'Iter : {iter} / {args.n_iters_dataloader}')
        for idx, batch in enumerate(test_loader):
            print(f'{idx} / {len(test_loader)}')
            data = dict_to_device(batch, device, to_skip=['strokes', 'time_steps'])
            targets = data['strokes_seq']
            starting_point = batch['canvas'][0].permute(1,2,0).cpu().numpy()
            starting_point = 0.5 * (starting_point + 1)
            bs = targets.size(0)
            # ======= Predict   ===========
            PT_params, _ = PT.main(data['img'], data['canvas'], strokes_ctx=data['strokes_ctx'])
            IP_params_new = new_IP.generate(data, no_z=True)
            IP_params_old = old_IP.generate(data, no_z=True)

            # ======= Masked L2 =======
            compute_metrics(preds=PT_params,
                            y=data['strokes_seq'],
                            renderer=renderer,
                            canvas_st=starting_point,
                            img_ref=data['img'],
                            meters=baseline_meters)

            compute_metrics(preds=IP_params_new.cpu().numpy(),
                            y=data['strokes_seq'],
                            renderer=renderer,
                            canvas_st=starting_point,
                            img_ref=data['img'],
                            meters=model_new_meters)

            compute_metrics(preds=IP_params_old.cpu().numpy(),
                            y=data['strokes_seq'],
                            renderer=renderer,
                            canvas_st=starting_point,
                            img_ref=data['img'],
                            meters=model_old_meters)

            # ======= LPIPS  =========
            compute_lpips(data=data, model=new_IP, meter=model_new_meters, n=args.n_samples, st=starting_point)
            compute_lpips(data=data, model=old_IP, meter=model_old_meters, n=args.n_samples, st=starting_point)

    baseline_meters.update({'fd_res' : baseline_meters['fd'].compute_fd()})
    model_new_meters.update({'fd_res' : model_new_meters['fd'].compute_fd()})
    model_old_meters.update({'fd_res' : model_old_meters['fd'].compute_fd()})

    print(f'*******************************************************************\n'
                 f'                 BASELINE              \t||\t          OLD    \t||\t           NEW\n'
                 f'FD (All)         Baseline : {baseline_meters["fd_res"]["all"] :.4f} \t||\t Our : {model_old_meters["fd_res"]["all"] :.4f} \t||\t New : {model_new_meters["fd_res"]["all"] :.4f}\n'
                 f'FD (Position)    Baseline : {baseline_meters["fd_res"]["position"] :.4f} \t||\t Our : {model_old_meters["fd_res"]["position"] :.4f}   \t||\t New : {model_new_meters["fd_res"]["position"] :.4f}\n'
                 f'FD (Color)       Baseline : {baseline_meters["fd_res"]["color"] :.4f} \t||\t Our : {model_old_meters["fd_res"]["color"] :.4f}  \t||\t New : {model_new_meters["fd_res"]["color"] :.4f}\n'
                 f'Masked L2        Baseline : {baseline_meters["masked_l2"].avg :.4f} \t||\t Our : {model_old_meters["masked_l2"].avg  :.4f}  \t||\t New : {model_new_meters["masked_l2"].avg  :.4f}\n'
                 f'MaskedL2/Area    Baseline : {baseline_meters["masked_l2_over_area"].avg :.5f} \t||\t Our : {model_old_meters["masked_l2_over_area"].avg :.5f}  \t||\t New : {model_new_meters["masked_l2_over_area"].avg :.5f}\n'
                 f'LPIPS            Our : {model_old_meters["lpips"].avg :.4f} \t||\t New {model_new_meters["lpips"].avg :.4f} ')

    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, 'baseline.pkl'), 'wb') as f:
        pkl.dump(baseline_meters, f)
    with open(os.path.join(args.output_path, 'model.pkl'), 'wb') as f:
        pkl.dump(model_new_meters, f)
    with open(os.path.join(args.output_path, 'old_model.pkl'), 'wb') as f :
        pkl.dump(model_old_meters, f)
