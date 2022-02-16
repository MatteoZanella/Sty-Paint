import argparse
import os

from model.utils.utils import dict_to_device
from model.utils.parse_config import ConfigParser
from model import build_model
from model.dataset import EvalDataset

from dataset_acquisition.decomposition.painter import Painter
from dataset_acquisition.decomposition.utils import load_painter_config
import torch
import numpy as np
import cv2
from collections import OrderedDict
import matplotlib.pyplot as plt

from evaluation.paint_transformer.model import PaintTransformer as PaddlePT
from evaluation.paint_transformer.torch_implementation import PaintTransformer
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


def produce_visuals(params, ctx, renderer, st, seq=None) :
    fg, alpha = renderer.inference(params, canvas_start=st)
    _, alpha_ctx = renderer.inference(ctx)
    cont = visualize(fg, alpha, alpha_ctx)
    if seq is not None:
        _, alpha_seq = renderer.inference(seq)
        alpha_seq = ((alpha_seq.sum(0) * 255)[:, :, None]).astype('uint8')
        contours_seq, _ = cv2.findContours(alpha_seq, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cont = cv2.drawContours(cont, contours_seq, -1, (0, 0, 255), 1)

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
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--checkpoint_baseline", type=str,
                        default='/home/eperuzzo/PaintTransformerPaddle/inference/paint_best.pdparams')

    parser.add_argument("--config", default='configs/eval/eval_v2.yaml')
    parser.add_argument("--output_path", type=str, default='./results/')
    args = parser.parse_args()


    # Create config
    c_parser = ConfigParser(args.config, isTrain=False)
    c_parser.parse_config()
    config = c_parser.get_config()
    print(config)

    # Create dataset_acquisition
    # device = config["device"]

    # Test
    dataset_test = EvalDataset(config, isTrain=False)
    print(f'Test : {len(dataset_test)} samples')

    # ======= Create Models ========================
    # Renderer (Stylized Neural Painting)
    render_config = load_painter_config(config["renderer"]["painter_config"])
    renderer = Painter(args=render_config)

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

    models = dict(
        our = model,
        #baseline = baseline
    )

    files = {
        'Bengal_192' : 40,
        'american_bulldog_115' : 191,
        'Bombay_153' : 34,
        'german_shorthaired_106' : 120,
        'Abyssinian_55' : 480,
        'Abyssinian_117':590,
        'Abyssinian_206' : 130,
        'Abyssinian_120' : 10,
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
        reference = 'Reference',
        original='Original',
        our_w_z ='Our w/ z',
        our_wo_z = 'Our w/o z',
        snp = 'SNP',
        pt='PT',
        )

    for filename, ts in files.items():
        batch, _, _ = dataset_test.sample(filename, ts)

        # batch =
        data = dict_to_device(batch, to_skip=['strokes', 'time_steps'])
        targets = data['strokes_seq']
        starting_point = batch['canvas'][0].permute(1, 2, 0).cpu().numpy()
        img = (batch['img'][0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')

        n_iters = 5
        visuals = dict()

        our_predicitons = dict(
            our_w_z=dict(),
            our_wo_z=dict()
        )
        ## Our model
        for n in range(n_iters):
            with torch.no_grad():
                preds = model(data, sample_z=True, seq_length=8)
                our_predicitons['our_w_z'].update({n : preds["fake_data_encoded"].cpu().numpy()})
                our_predicitons['our_wo_z'].update({n : preds["fake_data_random"].cpu().numpy()})

        ## Baseline model
        baseline_predictions = baseline.generate(data)
        snp_predicitons = renderer.generate(data)
        if torch.is_tensor(baseline_predictions) :
            baseline_predictions = baseline_predictions.cpu().numpy()
            snp_predicitons = snp_predicitons.cpu().numpy()

        visuals.update({'pt' : produce_visuals(baseline_predictions, batch['strokes_ctx'], renderer, starting_point)})
        visuals.update({'snp' : produce_visuals(snp_predicitons, batch['strokes_ctx'], renderer, starting_point)})
        visuals.update({'original' : produce_visuals(batch['strokes_seq'], batch['strokes_ctx'], renderer, starting_point)})
        visuals.update({'reference' : img})

        for name, tmp in our_predicitons.items():
            visuals.update({name : dict()})
            for n, preds in tmp.items():
                visuals[name].update({n : produce_visuals(preds, batch['strokes_ctx'], renderer, starting_point, seq=None)})

         # Shaw and save
        fig, axs = plt.subplots(n_iters, 6, figsize=(30,30), gridspec_kw = {'wspace':0, 'hspace':0})

        for n in range(n_iters):
            ii = 0
            for key, title in key_to_title.items():
                if not key.startswith('our'):
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
        plt.savefig(os.path.join(args.output_path, f'{filename}_preds.jpg'), bbox_inches='tight')