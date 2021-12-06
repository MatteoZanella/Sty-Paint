import argparse
import os

from model.utils.utils import dict_to_device
from model.utils.parse_config import ConfigParser
from model import model, model_2_steps
from model.dataset import EvalDataset

from dataset_acquisition.decomposition.painter import Painter
from dataset_acquisition.decomposition.utils import load_painter_config
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__' :
    # Extra parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--img_name", type=str, default='Abyssinian_206')
    parser.add_argument("--L", type=int, default=8, help='Number of strokes that will be predicted')
    parser.add_argument("--no_z", action='store_false', help='set the flag to USE z')
    parser.add_argument("--config", default='/home/eperuzzo/brushstrokes-generation/configs/train/sibiu_config.yaml')
    parser.add_argument("--n_iters", default=30, type=int, help='Number of iterations to generate strokes')
    parser.add_argument("--output_path", type=str, default='/home/eperuzzo/our_video/')
    parser.add_argument("--checkpoint_baseline", type=str,
                        default='/home/eperuzzo/PaintTransformer/inference/paint_best.pdparams')
    args = parser.parse_args()

    print(args.no_z)
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

    net = model_2_steps.InteractivePainter(config)
    net.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])
    net.to(config["device"])
    net.eval()

    os.makedirs(args.output_path, exist_ok=True)
    T =  transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(), ])

    filename = args.img_name
    t_start = 50
    batch = dataset_test.sample(filename, t_start)
    ref_img = batch['img']
    starting_point = batch['canvas'][0].permute(1, 2, 0).cpu().numpy()

    n_iters = args.n_iters
    predictions = dict()
    results = dict()

    output_path = os.path.join(args.output_path, args.img_name)
    os.makedirs(output_path, exist_ok=True)

    plt.imsave(os.path.join(output_path, f'frame_000.jpg'), starting_point)
    ctx = torch.empty((1, config["dataset"]["context_length"], config["model"]["n_strokes_params"]))

    for n in range(n_iters):

        data = dict_to_device(batch, device, to_skip=['strokes', 'time_steps'])
        preds = net.generate(data, no_z=args.no_z, L=args.L)
        this_frame, _ = renderer.inference(preds.cpu().numpy(), canvas_start=starting_point)
        plt.imsave(os.path.join(output_path, f'frame_{str(n+1).zfill(3)}.jpg'), this_frame)
        starting_point = this_frame

        # Update context
        ctx = batch['strokes_ctx'] # copy
        ctx = torch.roll(ctx, shifts=args.L, dims=1)   # shift the context by the number of predicted strokes
        ctx[:, :args.L, :] = torch.flip(preds, dims=(1,))  # FIFO

        if args.no_z:
            batch = {
                'img' : ref_img,
                'canvas' : torch.tensor((this_frame)).unsqueeze(0).permute(0, 3, 1, 2),
                'strokes_ctx' : ctx,
                'strokes_seq' : torch.randn_like(preds)
            }
        else:
            t_start += 8
            batch = dataset_test.sample(filename, t_start)
            batch['strokes_ctx'][:, -8:, :] = preds
