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
import matplotlib.pyplot as plt
import evaluation.tools as etools

from evaluation.paint_transformer.model import PaintTransformer as PaddlePT
from evaluation.tools import render_frames
import warnings
warnings.filterwarnings("ignore")

from einops import rearrange, repeat
import torch.nn.functional as F
import imageio
import random
# import paddle

from dataset_acquisition.sorting.graph import Graph
from dataset_acquisition.sorting.utils import load_segmentation, StrokesLoader


files = {
    'american_bulldog_115' : 191,
    'Maine_Coon_264' : 70,
    'beagle_162' : 75,
    'german_shorthaired_106' : 91,
    'Bombay_153' : 34,
    'beagle_62' : 675,
    'Bengal_34' : 678,
    'British_Shorthair_130' : 640,
    'english_cocker_spaniel_112' : 290,
    'great_pyrenees_44' : 375,
    'staffordshire_bull_terrier_81' : 120}

# files = {
#     'Bengal_192' : 30,
#     'american_bulldog_115' : 191,
#
#     'german_shorthaired_106' : 91,
#     'Abyssinian_55' : 48,
#     'Abyssinian_206' : 50,
#     'Maine_Coon_264' : 70,
#     'beagle_67' : 260,
#     'beagle_125' : 75,
#     'beagle_32' : 225,
#     'basset_hound_81' : 65,
#     'boxer_27' : 60,
#     'Bengal_85' : 72,
#     'Egyptian_Mau_74' : 85,
#     'beagle_162' : 75,
#     'staffordshire_bull_terrier_81' : 92}
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

def compute_diff(x):

    if torch.is_tensor(x):
        l1 = torch.abs(torch.diff(x, dim=1)).mean(dim=-1)
        l2 = torch.pow(torch.diff(x, dim=1), 2).mean(dim=-1)
    else:
        l1 = np.abs(np.diff(x, axis=1)).mean(axis=-1)
        l2 = np.square(np.diff(x, axis=1)).mean(axis=-1)

    return l2[0]

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


def save_histogram(our,
               gt,
               baseline,
               path,
               filename):


    our_color = compute_diff(our[:, :, 5:])
    our_pos = compute_diff(our[:, :, :2])

    gt_color = compute_diff(gt[:, :, 5:])
    gt_pos = compute_diff(gt[:, :, :2])

    baseline_color = compute_diff(baseline[:, :, 5:])
    baseline_pos = compute_diff(baseline[:, :, :2])


    pos_bins = (0, gt_pos.max().item())
    col_bins = (0, gt_color.max().item())

    f = plt.figure(figsize=(20,15))

    plt.subplot(3,2,1)
    plt.hist(gt_pos.cpu().numpy(), bins=50, range=pos_bins, density=True)
    plt.title('GT pos')

    plt.subplot(3,2,2)
    plt.hist(gt_color.cpu().numpy(), bins=50, range=col_bins, density=True)
    plt.title('GT color')

    plt.subplot(3, 2, 3)
    plt.hist(our_pos, bins=50, range=pos_bins, density=True)
    plt.title('Our pos')

    plt.subplot(3, 2, 4)
    plt.hist(our_color, bins=50, range=col_bins, density=True)
    plt.title('Our color')

    plt.subplot(3, 2, 5)
    plt.hist(baseline_pos, bins=50, range=pos_bins, density=True)
    plt.title('Baseline pos')

    plt.subplot(3, 2, 6)
    plt.hist(baseline_color, bins=50, range=col_bins, density=True)
    plt.title('Baseline color')

    plt.savefig(os.path.join(path, filename + 'hist.png'))
    plt.close(f)


def compute_img_l2(gt,
                    our,
                    baseline,
                    data,
                    renderer,
                    path,
                    filename):

    frames_gt= render_frames(gt, data, renderer=renderer)
    frames_our= render_frames(our, data, renderer=renderer)
    frames_baseline= render_frames(baseline, data, renderer=renderer)

    R = data['img'][0].permute(1,2,0).cpu().numpy()
    C = data['canvas'][0].permute(1,2,0).cpu().numpy()

    base = np.square(R - C).mean().item()

    def cum_difference(ref, F, base):
        result = [base]
        for i in range(F.shape[1]):
            result.append(np.square(ref - F[0, i]).mean().item())
        return result

    gt_scores = cum_difference(R, frames_gt['frames'], base)
    our_scores = cum_difference(R, frames_our['frames'], base)
    baseline_scores = cum_difference(R, frames_baseline['frames'], base)

    f = plt.figure(figsize=(10,10))
    plt.plot(gt_scores, label='GT')
    plt.plot(our_scores, label='Our')
    plt.plot(baseline_scores, label='Baseline')
    plt.legend()

    plt.savefig(os.path.join(path, filename + 'loss_img.png'))
    plt.close(f)




def compute_cost(x, filename):
    _, L, _ = x.shape

    st = StrokesLoader(path='')
    st.strokes = x
    st.num_strokes = L

    # Load segmentation map
    sm = load_segmentation(os.path.join('/home/eperuzzo/oxford_dataset/annotations/trimaps/', filename + '.png'),
                           256)

    feat = st.add_segmentation_saliency(sm, 256)
    g = Graph(range(0, L), feat)
    g.set_weights({'color' : 1, 'area' : 1, 'pos' : 0.5, 'class' : 0, 'sal' : 0})

    C = []
    for i in range(0, L-1) :
        C.append(g.compute_metric(i, i + 1))

    print(sum(C) / len(C))

    return C




def combine_gifs(path, tot):
    # Create reader object for the gif
    gif1 = imageio.get_reader(os.path.join(path, f'original_video_{tot}.gif'))
    gif2 = imageio.get_reader(os.path.join(path, f'our_video_{tot}.gif'))
    gif3 = imageio.get_reader(os.path.join(path, f'baseline_video_{tot}.gif'))

    # If they don't have the same number of frame take the shorter
    number_of_frames = min(gif1.get_length(), gif2.get_length(), gif3.get_length())

    # Create writer object
    new_gif = []#imageio.get_writter(os.path.join(path, f'output_{tot}.gif'))

    for frame_number in range(number_of_frames) :
        img1 = gif1.get_next_data()
        img2 = gif2.get_next_data()
        img3 = gif3.get_next_data()
        # here is the magic
        new_image = np.hstack((img1, img2, img3))
        new_gif.append(new_image)

    gif1.close()
    gif2.close()
    gif3.close()
    imageio.mimsave(os.path.join(path, f'output_{tot}.gif'), new_gif, duration=0.1)
    os.remove(os.path.join(path, f'original_video_{tot}.gif'))
    os.remove(os.path.join(path, f'our_video_{tot}.gif'))
    os.remove(os.path.join(path, f'baseline_video_{tot}.gif'))
    #new_gif.close()


def predict(net, batch, renderer, n_iters=5, L=8, is_our=True):

    strokes = []
    drawn_strokes = 0
    starting_point = batch['canvas'][0].permute(1,2,0).cpu().numpy()

    tmp_batch = batch.copy()
    for n in range(n_iters) :
        clamp = 1
        data = dict_to_device(batch, to_skip=['strokes', 'time_steps'])
        # Predict
        if is_our:
            with torch.no_grad() :
                preds = net(data, sample_z=True, seq_length=L)["fake_data_random"]
        else:
            preds = net.generate(data)

        if not torch.is_tensor(preds):
            preds = torch.tensor(preds)

        preds = etools.check_strokes(preds, clamp_wh=clamp)
        # Append and render
        strokes.append(preds.cpu().numpy())
        this_frame, _ = renderer.inference(preds.cpu().numpy(), canvas_start=starting_point)
        starting_point = this_frame

        # Update context
        ctx = batch['strokes_ctx']  # copy
        diff = 10 - L
        ctx = torch.roll(ctx, shifts=diff, dims=1)  # shift the context by the number of predicted strokes
        ctx[:, diff :, :] = preds[:, :, :]
        tmp_batch.update(
            {'canvas' : torch.tensor((this_frame), dtype=batch['img'].dtype).unsqueeze(0).permute(0, 3, 1, 2),
             'strokes_ctx' : ctx,
             'strokes_seq' : torch.randn_like(preds)}
        )
        drawn_strokes += L

    strokes = np.concatenate(strokes, axis=1)
    return strokes



def main(args, exp_name):

    # Seed
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # paddle.seed(seed)
    torch.backends.cudnn.benchmark = True

    # Create config
    c_parser = ConfigParser(args.config, isTrain=False)
    c_parser.parse_config()
    config = c_parser.get_config()
    print(config)

    # Test
    dataset_test = EvalDataset(config, isTrain=False)
    print(f'Test : {len(dataset_test)} samples')
    # ======= Create Models ========================
    # Renderer (Stylized Neural Painting)
    render_config = load_painter_config(config["renderer"]["painter_config"])
    renderer = Painter(args=render_config)

    # Loop over files
    checkpoint_path = os.path.join(args.checkpoint_base, exp_name, 'latest.pth.tar')
    output_path = os.path.join(args.output_path, exp_name)

    # load checkpoint, update model config based on the stored config
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    config.update(dict(model=ckpt["config"]["model"]))

    model = build_model(config)
    msg = model.load_state_dict(ckpt["model"], strict=False)
    print(f'==> Loading model form {checkpoint_path}, with : {msg}')
    model.cuda()
    model.eval()

    # baseline Paint Transformer
    baseline = PaddlePT(model_path=args.checkpoint_baseline, config=render_config)

    os.makedirs(output_path, exist_ok=True)

    n_iters = args.n_iters
    L = 8
    tot = n_iters * L
    for filename, ts in files.items() :
        print(f'==> Processing img : {filename}')

        os.makedirs(os.path.join(output_path, filename), exist_ok=True)

        batch, _, original_seq = dataset_test.sample(filename, ts, tot=tot)

        data = dict_to_device(batch, to_skip=['strokes', 'time_steps'])
        targets = data['strokes_seq']
        starting_point = batch['canvas'][0].permute(1, 2, 0).cpu().numpy()
        img = (batch['img'][0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')

        # Our
        our_prediction = predict(net=model, batch=batch, renderer=renderer, is_our=True, n_iters=args.n_iters)
        res_our = produce_visuals(our_prediction, batch['strokes_ctx'], renderer, starting_point)
        renderer.inference(our_prediction,
                           os.path.join(output_path, filename, f'our_video_{tot}'),
                           save_video=False,
                           save_jpgs=False,
                           save_gif=True,
                           canvas_start=starting_point)

        # Baseline
        baseline_preds = predict(net=baseline, batch=batch, renderer=renderer, is_our=False, n_iters=args.n_iters)
        res_baseline = produce_visuals(baseline_preds, batch['strokes_ctx'], renderer, starting_point)
        renderer.inference(baseline_preds,
                           os.path.join(output_path, filename, f'baseline_video_{tot}'),
                           save_video=False,
                           save_jpgs=False,
                           save_gif=True,
                           canvas_start=starting_point)

        # Reference
        original = produce_visuals(original_seq, batch['strokes_ctx'], renderer, starting_point)
        renderer.inference(original_seq,
                           os.path.join(output_path, filename, f'original_video_{tot}'),
                           save_video=False,
                           save_jpgs=False,
                           save_gif=True,
                           canvas_start=starting_point)

        ## plot difference
        save_histogram(our=our_prediction,
                       gt=original_seq,
                       baseline=baseline_preds,
                       path=output_path,
                       filename=filename)

        '''
        compute_img_l2(gt=original_seq,
                       our=our_prediction,
                       baseline=baseline_preds,
                       data= data,
                       renderer=renderer,
                       path=output_path,
                       filename=filename)
        '''
        ##
        print('Original')
        compute_cost(original_seq, filename=filename)
        print('Our')
        compute_cost(our_prediction, filename=filename)
        print('Baseline')
        compute_cost(baseline_preds, filename=filename)


        fig, axs = plt.subplots(1, 4, figsize=(30, 10), gridspec_kw={'wspace' : 0, 'hspace' : 0})
        images = [img, original, res_our, res_baseline]
        title = ['Img', 'Original', 'Our', 'Baseline']
        for ii in range(len(images)) :
            axs[ii].imshow(images[ii])
            axs[ii].set_title(title[ii])
            axs[ii].axis('off')
            axs[ii].set_xticklabels([])
            axs[ii].set_yticklabels([])
            axs[ii].set_aspect('equal')
        plt.savefig(os.path.join(output_path, filename, f'result_{tot}.jpg'), bbox_inches="tight")
        plt.close(fig)

        # Load gifs and combine in a unique one
        combine_gifs(path=os.path.join(output_path, filename), tot=tot)


if __name__ == '__main__' :
    # Extra parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_base", type=str, required=True)
    parser.add_argument("--checkpoint_baseline", type=str,
                        default='/home/eperuzzo/PaintTransformerPaddle/inference/paint_best.pdparams')

    parser.add_argument('-n', '--n_iters', default=8, type=int)
    parser.add_argument('-l', '--seq_length', default=8, type=int)

    parser.add_argument("--config", default='configs/eval/eval_v2.yaml')
    parser.add_argument("--output_path", type=str, default='/home/eperuzzo/seq_results/')
    args = parser.parse_args()



    # List
    experiments = os.listdir(args.checkpoint_base)
    print(f'Experiments to evaluate : {experiments}')

    for exp in experiments:
        print(f'Processing: {exp}')
        main(args, exp_name=exp)


