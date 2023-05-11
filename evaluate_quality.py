import argparse
from model.utils.parse_config import ConfigParser
import random
import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import pickle

from torch.utils.data import DataLoader
from model.networks.light_renderer import LightRenderer

from dataset_acquisition.decomposition.painter import Painter
from dataset_acquisition.decomposition.utils import load_painter_config

from model.utils.parse_config import ConfigParser
from model import build_model
from model.dataset import StrokesDataset, StylizedStrokesDataset
from model.dataloader import DataLoaderWrapper, collate_strokes
import evaluation.tools as etools

class FakeDataset():
    def __init__(self, config) -> None:
        self.config = config

class FakeDataloader():
    def __init__(self, config) -> None:
        self.dataset = FakeDataset(config)

def main(args):
    # Seed
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

    c_parser = ConfigParser(args.config, isTrain=False)
    c_parser.parse_config()
    config = c_parser.get_config()
    print(config)

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    config.update(dict(model=ckpt["config"]["model"]))
    config['model']['context_encoder']['use_style_efdm'] = True
    config['model']['context_encoder']['use_style_tokens'] = False
    model = build_model(config)
    msg = model.load_state_dict(ckpt["model"], strict=False)
    print(f'==> Loading model form {args.checkpoint}, with : {msg}')
    model.to(config['device'])
    model.eval()

    light_renderer = LightRenderer(config["stylization"]["brush_paths"], config['dataset']['resize'], batch_size=config["stylization"]["renderer_batch_size"])
    
    render_config = load_painter_config(config["renderer"]["painter_config"])
    render_config.gpu_id = config["gpu_id"]
    renderer = Painter(args=render_config)

    CONTENT_FOLD_PATH = '../efdm/data_snp/content/'
    STYLE_FOLD_PATH = '../efdm/data_snp/style/'

    img_size =  config["stylization"]["resize"]
    img_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(), ])
    
    wrapper = DataLoaderWrapper(FakeDataloader(config))

    context_length = config['dataset']['context_length']
    sequence_length = config['dataset']['sequence_length']

    os.makedirs(args.output_path, exist_ok=True)

    for content_dir in os.listdir(CONTENT_FOLD_PATH)[-2:]:
        batch = []
        content_path = os.path.join(CONTENT_FOLD_PATH, content_dir, f"{content_dir}.jpg")
        content = img_transform(Image.open(content_path).convert('RGB'))
        out_cont_path = os.path.join(args.output_path, content_dir)
        os.makedirs(out_cont_path, exist_ok=True)
        # Strokes
        raw_strokes = np.load(os.path.join(CONTENT_FOLD_PATH, content_dir, 'strokes_params.npz'))
        color = 0.5 * (raw_strokes['x_color'][:, :, :3] + raw_strokes['x_color'][:, :, 3:])
        strokes = np.concatenate([raw_strokes['x_ctt'], color], axis=-1)
        strokes = torch.tensor(strokes, dtype=torch.float).squeeze(0)
        file = os.path.join(CONTENT_FOLD_PATH, content_dir, 'lkh_col1_area1_pos0.5_cl2_sal0.pkl')
        with open(file, 'rb') as f:
            idx = pickle.load(f)
        strokes = strokes[idx]

        for style_file in os.listdir(STYLE_FOLD_PATH)[-3:-2]:
            style_path = os.path.join(STYLE_FOLD_PATH, style_file)
            style = img_transform(Image.open(style_path).convert('RGB'))
            if args.sample:
                steps = [100] * 15
            else:
                steps = [50, 100, 200, 500]
            for i, t in enumerate(steps):
                t_C = t - context_length
                t_T = t + sequence_length
                sample_strokes = strokes[:t_T, :]
                batch.append(
                    {'time_steps': torch.tensor([t_C, t, t_T]).to(config['device']),
                    'strokes': sample_strokes.to(config['device']),
                    'content': content.to(config['device']),
                    'style': style.to(config['device']),
                    't': f"{t}_{i}" if args.sample else t,
                    'sty_name': style_file.split('.')[0]
                    })
        batch = collate_strokes(batch)
        t, sty_name = batch['t'], batch['sty_name']
        batch = wrapper.style_preprocess(batch)
        batch['t'] = t
        batch['sty_name'] = sty_name
        
        predictions = model.generate(batch)["fake_data_random"]
        predictions = etools.check_strokes(predictions)  # clamp in range [0,1]
        
        predictions = predictions.unsqueeze(1).cpu().numpy()
        targets = batch['strokes_seq'].unsqueeze(1).cpu() # ???
        contexts = batch['strokes_ctx'].unsqueeze(1).cpu()
        canvases = batch['canvas'].permute(0, 2, 3, 1).cpu().numpy()
        steps = batch['t']
        sty_names = batch['sty_name']

        for prediction, canvas, context, target, t, sty_name in zip(predictions, canvases, contexts, targets, steps, sty_names):
            visuals = etools.produce_visuals(
                prediction, renderer, canvas, 
                ctx=context, seq=None if args.sample else target)
            out_path = os.path.join(out_cont_path, sty_name)
            if args.sample:
                out_path = os.path.join(out_path, 'sample')
            os.makedirs(out_path, exist_ok=True)
            Image.fromarray(visuals).save(os.path.join(out_path, f'{t}.png'))

if __name__ == '__main__':
    # Extra parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=False)
    parser.add_argument("--config", default='configs/eval/eval_quality.yaml')
    parser.add_argument("--output_path", type=str, default='eval_quality/')
    parser.add_argument("--sample", action='store_true', default=False)
    args = parser.parse_args()

    # Run
    main(args)
