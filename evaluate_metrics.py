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
from model.networks.light_renderer import LightRenderer

from model.utils.utils import dict_to_device, AverageMetersDict
from model.utils.parse_config import ConfigParser
from model import build_model
from model.dataset import StrokesDataset, StylizedStrokesDataset
from model.dataloader import DataLoaderWrapper, collate_strokes

from dataset_acquisition.decomposition.painter import Painter
from dataset_acquisition.decomposition.utils import load_painter_config

from evaluation.metrics import StyleLoss, StrokesStyleDistance, WassersteinDistance, FSD, StrokeColorL2, DTW, compute_color_difference
from evaluation.fvd import FVD
from evaluation.paint_transformer.model import PaintTransformer as PaddlePT
import evaluation.tools as etools


class EvalModel:
    def __init__(self, renderer, config):
        super(EvalModel, self).__init__()
        self.renderer = renderer
        self.stylization = config["stylization"]["apply"]
        if self.stylization:
            img_size = config['dataset']['resize']
            brush_paths = config["stylization"]["brush_paths"]
            batch_size = config["stylization"]["renderer_batch_size"]
            self.light_renderer = LightRenderer(brush_paths, img_size, batch_size=batch_size)
        self.stylize_img = "stylize_img" not in config["stylization"] or config["stylization"]["stylize_img"]

        self.scl2Metric = StrokeColorL2()
        self.wdMetric = WassersteinDistance()
        self.dtwMetric = DTW()
        self.ssdMetric = StrokesStyleDistance()
        if self.stylization:
            vgg_weights = config['stylization']["vgg_weights"]
            self.slMetric = StyleLoss(vgg_weights=vgg_weights, device=config['device'])

    def __call__(self, data, net, metric_logger, is_our=False):
        targets = data['strokes_seq'].cpu()
        context = data['strokes_ctx'].cpu()
        if self.stylization:
            canvas = data['canvas'].cpu()
            style = data['style'].cpu()
        bs = targets.size(0)
        img_target = data['img'] if self.stylize_img else data['img_target']

        if is_our:
            predictions = net.generate(data)["fake_data_random"]
        else:
            predictions = net.generate(data)
        predictions = etools.check_strokes(predictions)  # clamp in range [0,1]

        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()

        # Visual rendering
        visuals = etools.render_frames(predictions, data, self.renderer)
        if self.stylization:
            pred_canvas = etools.render_canvas_light(predictions, canvas, self.light_renderer)
            target_canvas = etools.render_canvas_light(targets, canvas, self.light_renderer)

        # compute metrics
        color_diff_l1, color_diff_l2 = compute_color_difference(predictions)
        scl2 = self.scl2Metric(img_target, visuals['frames'], visuals['alphas'])
        wd = self.wdMetric(targets[:, :, :5], torch.tensor(predictions[:, :, :5]))
        dtw = self.dtwMetric(targets, predictions)
        ssd = self.ssdMetric(targets, predictions, ctx=context)
        if self.stylization:
            sl_context = self.slMetric(canvas, style)
            sl_pred = self.slMetric(pred_canvas, style)
            sl_target = self.slMetric(target_canvas, style)
            sc = self.slMetric.style_contrib(sl_context, sl_pred)
            sa = self.slMetric.style_accuracy(sl_target, sl_pred)

        # record metrics
        metrics = {
            'scl2': scl2.item(),
            'wd': wd.item(),
            'color_diff_l1' : color_diff_l1.item(),
            'color_diff_l2': color_diff_l2.item(),
            'dtw': dtw.item(),
            'ssd': ssd.item(),
            # 'ssd': ssd["ssd"].item(),
            # 'ssd_pos': ssd["ssd_pos"].item(),
            # 'ssd_color': ssd["ssd_color"].item(),
        }
        # Stylization metrics
        if self.stylization:
            metrics['sc'] = sc.item()
            metrics['sa'] = sa.item()
        metric_logger.update(metrics, bs)

        return predictions, visuals['frames']

def main(args):
    # Seed
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    paddle.seed(seed)
    torch.backends.cudnn.benchmark = True

    # gpus = tf.config.list_physical_devices('GPU')
    # tf.config.set_visible_devices(gpus[0], 'GPU')
    # tf.config.experimental.set_memory_growth(gpus[0], True)

    # Create config
    c_parser = ConfigParser(args.config, isTrain=False)
    c_parser.parse_config()
    config = c_parser.get_config()
    print(config)

    # Collate function
    collate_fn = collate_strokes if config["stylization"]["apply"] else None

    # Test
    if config["stylization"]["apply"]:
        dataset_test = StylizedStrokesDataset(config, isTrain=False)
    else:
        dataset_test = StrokesDataset(config, isTrain=False)
    test_loader = DataLoader(dataset=dataset_test,
                             batch_size=16,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=True,
                             collate_fn=collate_fn)
    test_loader = DataLoaderWrapper(test_loader)
    print(f'Test : {len(dataset_test)} samples')

    # ======= Create Models ========================
    # Renderer (Stylized Neural Painting)
    render_config = load_painter_config(config["renderer"]["painter_config"])
    render_config.gpu_id = config["gpu_id"]
    renderer = Painter(args=render_config)
    if args.use_snp2:
        snp_plus_config = copy.deepcopy(render_config)
        snp_plus_config.with_kl_loss = True
        snp_plus = Painter(args=snp_plus_config)

    # load checkpoint, update model config based on the stored config
    use_our = args.checkpoint is not None
    if use_our:
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        config.update(dict(model=ckpt["config"]["model"]))
        # config['model']['context_encoder']['use_style_efdm'] = True
        # config['model']['context_encoder']['use_style_tokens'] = False
        model = build_model(config)
        msg = model.load_state_dict(ckpt["model"], strict=False)
        print(f'==> Loading model form {args.checkpoint}, with : {msg}')
        model.to(config['device'])
        model.eval()

    # Output Path
    if use_our:
        exp_name = ckpt["config"]["train"]["logging"]["exp_name"]
        our_output_path = os.path.join(args.output_path, ckpt["config"]["train"]["logging"]["exp_name"])

    # Paint Transformer
    if args.use_pt:
        baseline = PaddlePT(model_path=args.pt_checkpoint, config=render_config)
    # ======= Metrics ========================
    fsdMetric = FSD()
    eval_model = EvalModel(renderer=renderer, config=config)
    original, ctx, our, pt, snp, snp2 = [], [], [], [], [], []
    visual_original, visual_ctx, visual_our, visual_pt, visual_snp, visual_snp2 = [], [], [], [], [], []

    # Average Meters
    eval_names = ['fsd', 'scl2', 'dtw', 'lpips', 'wd', 'color_diff_l1', 'color_diff_l2', 'ssd']
    if config["stylization"]["apply"]:
        eval_names.extend(['sc', 'sa'])
    
    if use_our:
        our_metrics = AverageMetersDict(names=eval_names)
    if args.use_pt:
        baseline_metrics = AverageMetersDict(names=eval_names)
    if args.use_snp:
        snp_metrics = AverageMetersDict(names=eval_names)
    if args.use_snp2:
        snp_plus_metrics = AverageMetersDict(names=eval_names)
    dataset_metrics = AverageMetersDict(names=['color_diff_l1', 'color_diff_l2'])

    # ======= Run ========================
    for iter in range(args.n_iters_dataloader):
        print(f'Iter : {iter + 1} / {args.n_iters_dataloader}')
        for idx, batch in enumerate(test_loader):
            print(f'{idx + 1} / {len(test_loader)}')
            # ======= Dataset Metrics ====
            ref_l1, ref_l2 = compute_color_difference(batch['strokes_seq'])
            dataset_metrics.update(dict(color_diff_l1=ref_l1.item(), color_diff_l2=ref_l2.item()),
                                   batch['strokes_seq'].shape[0])
            # ======= Predict   ===========
            # Our
            if use_our:
                our_predictions, our_vis = eval_model(data=batch, net=model, metric_logger=our_metrics, is_our=True)
            if args.use_pt:
                pt_predictions, pt_vis = eval_model(data=batch, net=baseline, metric_logger=baseline_metrics)
            if args.use_snp:
                snp_predictions, snp_vis = eval_model(data=batch, net=renderer, metric_logger=snp_metrics)
            if args.use_snp2:
                snp2_predictions, snp2_vis = eval_model(data=batch, net=snp_plus, metric_logger=snp_plus_metrics)

            # Add to the list
            if use_our:
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
            # Move to CPU the parameters
            strokes_seq = batch['strokes_seq'].cpu()
            strokes_ctx = batch['strokes_ctx'].cpu()
            # Original
            original.append(strokes_seq)
            ctx.append(strokes_ctx)
            orig_vis = etools.render_frames(strokes_seq, batch, renderer)
            ctx_vis = etools.render_frames(strokes_ctx, batch, renderer)
            visual_original.append(orig_vis['frames'])
            visual_ctx.append(ctx_vis['frames'])
    
    # Delete models
    if use_our:
        del model
    if args.use_pt:
        del baseline
    if args.use_snp:
        del renderer
    if args.use_snp2:
        del snp_plus

    # Aggregate metrics
    results = {}
    visual_original = np.concatenate(visual_original).astype(np.half)
    visual_ctx = np.concatenate(visual_ctx).astype(np.half)
    results.update(dataset_metrics.get_avg(header='dataset_'))

    if use_our:
        results.update(our_metrics.get_avg(header='our_'))
        our_fid = fsdMetric(generated=our, original=original, ctx=ctx)
        visual_our = np.concatenate(visual_our).astype(np.half)
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
        if use_our:
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
    os.makedirs(args.output_path, exist_ok=True)
    # with open(os.path.join(args.output_path, f'results.pkl'), 'wb') as f:
    #     pkl.dump(results, f)
    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['value'])
    if use_our:
        # folder = our_output_path
        # os.makedirs(folder, exist_ok=True)
        path = os.path.join(args.output_path, f'{exp_name}.csv')
        results_df[results_df.index.str.startswith('our')].to_csv(path)
    if args.use_pt:
        # folder = os.path.join(args.output_path,  'pt')
        # os.makedirs(folder, exist_ok=True)
        path = os.path.join(args.output_path, f'{exp_name}.csv')
        results_df[results_df.index.str.startswith('pt')].to_csv(path)
    if args.use_snp:
        folder = os.path.join(args.output_path,  'snp')
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, 'metrics.csv')
        results_df[results_df.index.str.startswith('snp')].to_csv(path)
    if args.use_snp2:
        folder = os.path.join(args.output_path,  'snp2')
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, 'metrics.csv')
        results_df[results_df.index.str.startswith('snp++')].to_csv(path)
    print('END')

if __name__ == '__main__':
    # Extra parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=False)
    parser.add_argument("--config", default='configs/eval/eval_oxford.yaml')

    parser.add_argument("--output_path", type=str, default='eval_metrics/')
    parser.add_argument("--pt_checkpoint", type=str,
                        default='/data1/mzanella/PaintTransformer/inference/paint_best.pdparams')
    parser.add_argument('--fvd', action='store_true')
    parser.add_argument("--n_samples", type=int, default=1,
                        help="number of samples to test lpips, diversity in generation")
    parser.add_argument("-n", "--n_iters_dataloader", default=5, type=int)  # 5
    parser.add_argument("--use-pt", action='store_true', default=False)
    parser.add_argument("--use-snp", action='store_true', default=False)
    parser.add_argument("--use-snp2", action='store_true', default=False)
    args = parser.parse_args()

    # Run
    main(args)
