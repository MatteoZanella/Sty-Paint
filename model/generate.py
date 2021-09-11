from dataset.decomposition.painter import Painter
from dataset.decomposition.utils import load_painter_config
import torch
from model.utils.utils import dict_to_device
import matplotlib.pyplot as plt
import os

class GenerateStorkes():

    def __init__(self, painter_config, output_path):

        args = load_painter_config(painter_config)
        self.device = torch.device(f'cuda:{args.gpu_id}')
        self.pt = Painter(args=args)

        self.output_path = output_path


    def generate_and_render(self, model, dataloader):
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                data = dict_to_device(data, self.device, to_skip=['strokes', 'time_steps'])
                generated_strokes = model.generate(data, L=8)
                generated_strokes = generated_strokes.detach().cpu().numpy()
                renders, _ = self.pt.inference(strokes=generated_strokes)
        return renders, generated_strokes

    def save_renders(self, render, filename, fmt='.jpg'):
        file = os.path.join(self.output_path, filename + fmt)
        plt.imsave(file, render)

