import argparse
import os

from model.utils.utils import dict_to_device
from model.utils.parse_config import ConfigParser
from model import model, model_2_steps
from model.dataset import EvalDataset

from dataset_acquisition.decomposition.painter import Painter
from dataset_acquisition.decomposition.utils import load_painter_config
import torch
import numpy as np
import cv2
from collections import OrderedDict
import matplotlib.pyplot as plt

from evaluation.paint_transformer.model import PaintTransformer as PaddlePT
import warnings
warnings.filterwarnings("ignore")

from einops import rearrange, repeat
import torch.nn.functional as F

def count_parameters(net) :
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def sample_color(params, imgs) :
    if not torch.is_tensor(params):
        params = torch.tensor(params)
    bs, n_strokes, _ = params.shape
    img_temp = repeat(imgs, 'bs ch h w -> (bs L) ch h w', L=n_strokes)
    grid = rearrange(params[:, :, :2], 'bs L p -> (bs L) 1 1 p')
    color = F.grid_sample(img_temp, 2 * grid - 1, align_corners=False)
    color = rearrange(color, '(bs L) ch 1 1 -> bs L ch', L=n_strokes)
    color = color.repeat(1,1,2)
    out_params = torch.cat((params.clone()[:, :, :5], color), dim=-1)
    return out_params.cpu().numpy()


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


def produce_visuals(params, ctx, renderer, st) :
    fg, alpha = renderer.inference(params, canvas_start=st)
    _, alpha_ctx = renderer.inference(ctx)
    cont = visualize(fg, alpha, alpha_ctx)

    return cont


def visualize(foreground, alpha, alpha_ctx) :
    tmp = ((alpha.sum(0) * 255)[:, :, None]).astype('uint8')
    contours, hierarchy = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    tmp_alpha = ((alpha_ctx.sum(0) * 255)[:, :, None]).astype('uint8')
    contours_ctx, _ = cv2.findContours(tmp_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    x = (np.copy(foreground) * 255).astype('uint8')
    res = cv2.drawContours(x, contours_ctx, -1, (255, 0, 0), 1)
    res = cv2.drawContours(res, contours, -1, (0, 255, 0), 1)
    return res


if __name__ == '__main__' :
    # Extra parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_1", type=str, required=True)
    parser.add_argument("--ckpt_2", type=str, required=True)
    parser.add_argument("--config", default='/home/eperuzzo/brushstrokes-generation/configs/train/sibiu_config.yaml')

    parser.add_argument("--output_path", type=str, default='/home/eperuzzo/eval_metrics/')
    parser.add_argument("--checkpoint_baseline", type=str,
                        default='/home/eperuzzo/PaintTransformer/inference/paint_best.pdparams')
    parser.add_argument("--n_samples_lpips", type=int, default=3,
                        help="number of samples to test lpips, diverstiy in generation")
    parser.add_argument("--n_iters_dataloader", default=1, type=int)
    args = parser.parse_args()


    # Create config
    c_parser = ConfigParser(args.config, isTrain=False)
    c_parser.parse_config(args)
    config = c_parser.get_config()
    print(config)

    # Create dataset_acquisition
    device = config["device"]

    # Test
    dataset_test = EvalDataset(config, isTrain=False)
    #test_loader = DataLoader(dataset=dataset_test, batch_size=16, shuffle=True, pin_memory=False)
    print(f'Test : {len(dataset_test)} samples')

    # ======= Create Models ========================
    # Renderer (Stylized Neural Painting)
    render_config = load_painter_config(config["render"]["painter_config"])
    renderer = Painter(args=render_config)

    net1 = model.InteractivePainter(config)
    net1.load_state_dict(torch.load(args.ckpt_1, map_location=device)["model"])
    net1.to(config["device"])
    net1.eval()

    net2 = model.InteractivePainter(config)
    net2.load_state_dict(torch.load(args.ckpt_2, map_location=device)["model"])
    net2.to(config["device"])
    net2.eval()

    models = dict(
        model = net1,
        model_two_steps = net2,
        baseline = PaddlePT(model_path=args.checkpoint_baseline, config=render_config))



    files = {
        'Bengal_192' : 100,
        'american_bulldog_115' : 191,
        'Bombay_153' : 34,
        'german_shorthaired_106' : 91,
        'Abyssinian_55' : 48,
        'Abyssinian_117':590,
        'Abyssinian_206' : 50,
        'Abyssinian_120' : 120,
        'beagle_62' : 260,}
        # 'beagle_27' : 273,
        # 'Maine_Coon_264' : 60,
        # 'beagle_67' : 260,
        # 'beagle_125' : 75,
        # 'beagle_32' : 225,
        # 'basset_hound_81' : 65,
        # 'boxer_27' : 60,
        # 'Bengal_85' : 72,
        # 'shiba_inu_123' : 50,
        # 'shiba_inu_191' : 60,
        # 'Egyptian_Mau_74': 85,
        # 'beagle_162' : 75,
        # 'Maine_Coon_123' : None,
        # 'shiba_inu_155' : 120,
        # 'Sphynx_244' : 45,
        # 'yorkshire_terrier_34' : None,
        # 'american_pit_bull_terrier_184' : 65,
        # 'saint_bernard_187' : 65,
        # 'staffordshire_bull_terrier_81' : 92,
        # 'Bombay_33' : None,
        # 'american_pit_bull_terrier_16' : None,
        # 'Bengal_154' : 56,
        # 'British_Shorthair_58' : None,
        # 'Egyptian_Mau_111' : 75,
        # 'Russian_Blue_82' : None,
        # 'Siamese_30' : None,
        # 'staffordshire_bull_terrier_4' : 45,
        # 'scottish_terrier_55' : None,
        # 'staffordshire_bull_terrier_169' : None,
        # 'staffordshire_bull_terrier_144' : None,
        # 'staffordshire_bull_terrier_119' : None,
        # 'wheaten_terrier_94' : None}


    os.makedirs(args.output_path, exist_ok=True)
    key_to_title = OrderedDict(
        reference='Reference',
        original='Original',
        baseline='Baseline',
        model='Our',
        model_sc='Our (sc)',
        model_two_steps='Our+'
        )

    for filename, ts in files.items():
        batch = dataset_test.sample(filename, ts)

        # batch =
        data = dict_to_device(batch, device, to_skip=['strokes', 'time_steps'])
        targets = data['strokes_seq']
        starting_point = batch['canvas'][0].permute(1, 2, 0).cpu().numpy()
        img = (batch['img'][0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')

        n_iters = 5
        predictions = dict()
        visuals = dict()

        for name, model in models.items() :
            predictions[name] = dict()
            if name == 'model':
                predictions['model_sc'] = dict()
            for n in range(n_iters):
                preds = model.generate(data)
                if torch.is_tensor(preds) :
                    preds = preds.cpu().numpy()
                predictions[name].update({n : preds})
                if name == 'model':
                    predictions['model_sc'].update({n : sample_color(predictions['model'][n], batch['img'])})

        visuals.update({'original' : produce_visuals(batch['strokes_seq'], batch['strokes_ctx'], renderer, starting_point)})
        visuals.update({'reference' : img})

        for name, tmp in predictions.items():
            visuals.update({name : dict()})
            for n, preds in tmp.items():
                visuals[name].update({n : produce_visuals(preds, batch['strokes_ctx'], renderer, starting_point)})

         # Shaw and save
        fig, axs = plt.subplots(n_iters, 6, figsize=(30,30), gridspec_kw = {'wspace':0, 'hspace':0})

        for n in range(n_iters):
            ii = 0
            for key, title in key_to_title.items():
                if key == 'original' or key == 'reference':
                    axs[n, ii].imshow(visuals[key])
                else:
                    axs[n, ii].imshow(visuals[key][n])
                if n == 0:
                    axs[n, ii].set_title(title)
                axs[n, ii].axis('off')
                axs[n, ii].set_xticklabels([])
                axs[n, ii].set_yticklabels([])
                axs[n, ii].set_aspect('equal')
                ii += 1
        plt.savefig(os.path.join(args.output_path, f'{filename}_preds.png'))
