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
from evaluation.tools import render_frames, produce_visuals
import warnings
warnings.filterwarnings("ignore")

from einops import rearrange, repeat
import torch.nn.functional as F
import imageio
import random
import copy
from scipy import stats
# import paddle

from dataset_acquisition.sorting.graph import Graph
from dataset_acquisition.sorting.utils import load_segmentation, StrokesLoader
import pandas as pd
'''
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
'''
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

def get_index(L, K=10) :
    ids = []
    for i in range(L) :
        for j in range(L) :
            if j < i :
                continue
            if (j > i + K) or i == j :
                continue
            else :
                ids.append([i, j])
    ids = np.array(ids)
    id0 = ids[:, 0]
    id1 = ids[:, 1]
    n = ids.shape[0]
    return id0, id1, n


def compute(x, attribute) :
    if attribute == 'pos' :
        x = x[:, :, :2]
    elif attribute == 'hw' :
        x = x[:, :, 2] * x[:, :, 3]
        x = x[:, :, None]
    elif attribute == 'theta' :
        x = x[:, :, 4][:, :, None]
    elif attribute == 'color' :
        x = x[:, :, 5 :]
    else :
        print('Unknown attribute')

    id0, id1, _ = get_index(L=x.shape[1])

    score = np.square(x[:, id0] - x[:, id1])
    return score.mean(axis=-1).reshape(-1)

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


'''
def produce_visuals(params, ctx, renderer, st) :
    fg, alpha = renderer.inference(params, canvas_start=st)
    _, alpha_ctx = renderer.inference(ctx)
    cont = visualize(fg, alpha, alpha_ctx)

    return cont


def visualize(foreground, alpha, alpha_ctx) :
    strokes_cnt, ctx_cnt = [], []
    for i in range(alpha.shape[0]):
        strokes_cnt.append(
            cv2.findContours(alpha[i][:, :, None].astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        )
    for i in range(alpha_ctx.shape[0]):
        ctx_cnt.append(
            cv2.findContours(alpha_ctx[i][:, :, None].astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        )

    x = (np.copy(foreground) * 255).astype('uint8')
    for cnt in ctx_cnt:
        cv2.drawContours(x, cnt, -1, (0, 0, 255), 1)
    for cnt in strokes_cnt:
        cv2.drawContours(x, cnt, -1, (255, 0, 0), 1)

    return x
'''

def save_histogram(inp, kde_pos, kde_col, path, filename):

    # Position
    pos = compute(inp, 'pos')

    x_pos = np.linspace(0, 0.3, num=200)
    y_pos = kde_pos(x_pos)

    f = plt.figure()
    plt.hist(pos, bins=50, density=True, label='prediction', color='red')
    plt.plot(x_pos, y_pos, label='reference', linewidth=2, color='blue')
    plt.title('Position')
    plt.ylim(bottom=0, top=1.6 * np.max(y_pos))
    plt.xlim(left=0, right=0.3)
    plt.xlabel('Relative Distance')
    plt.legend()
    plt.savefig(os.path.join(path, filename + '_pos.png'), bbox_inches='tight')
    plt.close(f)

    # Color
    color = compute(inp, 'color')

    x_col = np.linspace(0, 0.45, num=200)
    y_col = kde_col(x_col)

    f = plt.figure()
    plt.hist(color, bins=50, density=True, label='prediction', color='red')
    plt.plot(x_col, y_col, label='reference', linewidth=2, color='blue')
    plt.title('Color')
    plt.ylim(bottom=0, top=1.3 * np.max(y_col))
    plt.xlim(left=0, right=0.45)
    plt.xlabel('Relative Distance')
    plt.legend()
    plt.savefig(os.path.join(path, filename + '_col.png'), bbox_inches='tight')
    plt.close(f)

def main_histogram(our,
                    pt,
                    snp,
                    snp2,
                   kde_pos,
                   kde_col,
                    path):

    # Our
    save_histogram(our, kde_pos, kde_col, path, filename='our')
    save_histogram(pt, kde_pos, kde_col, path, filename='pt')
    save_histogram(snp, kde_pos, kde_col, path, filename='snp')
    save_histogram(snp2, kde_pos, kde_col, path, filename='snp2')



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

def combine_videos(path, tot):

    config = [f'original', f'our', f'pt', f'snp', 'snp2']
    video_readers = [imageio.get_reader(os.path.join(path, l + '_animated.mp4'), 'ffmpeg') for l in config]

    # Reading the frames
    F = {}

    for vd in video_readers :
        for idx, im in enumerate(vd) :
            if idx in F.keys() :
                F[idx].append(im)
            else :
                F[idx] = [im]

    # Creating the final video
    final_video = []
    for frame_id, tot_frames in F.items() :
        final_frame = np.concatenate((tot_frames[0], tot_frames[1], tot_frames[2]), axis=1)
        final_video.append(final_frame)

    # write final video
    out_name = os.path.join(path, 'animation_result.mp4')
    writer = imageio.get_writer(out_name, fps=10)
    for im in final_video :
        writer.append_data(im)
    writer.close()


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
                preds = net.generate(data)["fake_data_random"]
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

    print(drawn_strokes)
    strokes = np.concatenate(strokes, axis=1)
    return strokes


def main(args):

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

    snp_plus_config = copy.deepcopy(render_config)
    snp_plus_config.with_kl_loss = True
    snp_plus = Painter(args=snp_plus_config)

    # Loop over files
    exp_name = 'videos_db'
    checkpoint_path = os.path.join(args.model)
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
    pt = PaddlePT(model_path=args.checkpoint_baseline, config=render_config)

    os.makedirs(output_path, exist_ok=True)

    n_iters = args.n_iters
    L = 8
    tot = n_iters * L

    # Load and compute KDE
    gt_position = np.load('/home/eperuzzo/reference_position.np.npy')
    gt_color = np.load('/home/eperuzzo/reference_color.np.npy')

    kde_position = stats.gaussian_kde(gt_position, bw_method=0.1)
    kde_color = stats.gaussian_kde(gt_color, bw_method=0.1)


    files = pd.read_csv('/home/eperuzzo/config.csv')
    for idx, row in files.iterrows() :
        filename = row["filename"]
        ts = row["ts"]
        print(f'==> Processing img : {filename}')

        os.makedirs(os.path.join(output_path, filename), exist_ok=True)

        batch, _, original_seq = dataset_test.sample(filename, ts, tot=tot)

        data = dict_to_device(batch, to_skip=['strokes', 'time_steps'])
        targets = data['strokes_seq']
        starting_point = batch['canvas'][0].permute(1, 2, 0).cpu().numpy()
        img = (batch['img'][0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')

        # Our
        our_prediction = predict(net=model, batch=batch, renderer=renderer, is_our=True, n_iters=args.n_iters)
        res_our = produce_visuals(our_prediction,
                                  renderer=renderer,
                                  starting_canvas=starting_point,
                                  ctx=batch['strokes_ctx'])
        renderer.inference(our_prediction,
                           os.path.join(output_path, filename, f'our'),
                           save_video=True,
                           save_jpgs=False,
                           save_gif=False,
                           canvas_start=starting_point,
                           hilight=True)

        # Baseline
        pt_preds = predict(net=pt, batch=batch, renderer=renderer, is_our=False, n_iters=args.n_iters)
        res_pt = produce_visuals(pt_preds,
                                 renderer=renderer,
                                 starting_canvas=starting_point,
                                 ctx=batch['strokes_ctx'])

        renderer.inference(pt_preds,
                           os.path.join(output_path, filename, f'pt'),
                           save_video=True,
                           save_jpgs=False,
                           save_gif=False,
                           canvas_start=starting_point,
                           hilight=True)

        # SNP
        snp_preds = predict(net=renderer, batch=batch, renderer=renderer, is_our=False, n_iters=args.n_iters)
        res_snp = produce_visuals(snp_preds,
                                  renderer=renderer,
                                  starting_canvas=starting_point,
                                  ctx=batch['strokes_ctx'])
        renderer.inference(snp_preds,
                           os.path.join(output_path, filename, f'snp'),
                           save_video=True,
                           save_jpgs=False,
                           save_gif=False,
                           canvas_start=starting_point,
                           hilight=True)

        # SNP+
        snp2_preds = predict(net=snp_plus, batch=batch, renderer=renderer, is_our=False, n_iters=args.n_iters)
        res_snp2 = produce_visuals(snp2_preds,
                                   renderer=renderer,
                                   starting_canvas=starting_point,
                                   ctx=batch['strokes_ctx'])

        renderer.inference(snp2_preds,
                           os.path.join(output_path, filename, f'snp2'),
                           save_video=True,
                           save_jpgs=False,
                           save_gif=False,
                           canvas_start=starting_point,
                           hilight=True)


        main_histogram(our=our_prediction,
                       pt=pt_preds,
                       snp=snp_preds,
                       snp2=snp2_preds,
                       kde_pos=kde_position,
                       kde_col = kde_color,
                       path= os.path.join(output_path, filename))
        # Reference
        original = produce_visuals(original_seq,
                                   renderer=renderer,
                                   starting_canvas=starting_point,
                                   ctx=batch['strokes_ctx'])
        renderer.inference(original_seq,
                           os.path.join(output_path, filename, f'original'),
                           save_video=True,
                           save_jpgs=False,
                           save_gif=False,
                           canvas_start=starting_point,
                           hilight=True)

        flag = False
        if flag:
            fig, axs = plt.subplots(1, 5, figsize=(30, 10), gridspec_kw={'wspace' : 0, 'hspace' : 0})
            images = [img, original, res_our, res_pt, res_snp]
            title = ['Img', 'Original', 'Our', 'PT', 'SNP']
            for ii in range(len(images)) :
                axs[ii].imshow(images[ii])
                axs[ii].set_title(title[ii])
                axs[ii].axis('off')
                axs[ii].set_xticklabels([])
                axs[ii].set_yticklabels([])
                axs[ii].set_aspect('equal')
            plt.savefig(os.path.join(output_path, filename, f'result_{tot}.jpg'), bbox_inches="tight")
            plt.close(fig)
        else:
            images = [img, original, res_our, res_pt, res_snp, res_snp2]
            title = ['reference_img', 'original', 'our', 'pt', 'snp', 'snp2']

            for i, t in zip(images, title):
                plt.imsave(os.path.join(output_path, filename, f'{t}.png'), i)

        # Load gifs and combine in a unique one
        #combine_videos(path=os.path.join(output_path, filename), tot=tot)


if __name__ == '__main__' :
    # Extra parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint_baseline", type=str,
                        default='/home/eperuzzo/PaintTransformerPaddle/inference/paint_best.pdparams')

    parser.add_argument('-n', '--n_iters', default=8, type=int)
    parser.add_argument('-l', '--seq_length', default=8, type=int)

    parser.add_argument("--config", default='configs/eval/eval_v2.yaml')
    parser.add_argument("--output_path", type=str, default='/home/eperuzzo/seq_results/')
    args = parser.parse_args()


    main(args)


