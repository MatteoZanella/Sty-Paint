from dataset_acquisition.decomposition.painter import Painter
from dataset_acquisition.decomposition.utils import load_painter_config
import torch
from model.utils.utils import dict_to_device
import matplotlib.pyplot as plt
import os
import numpy as np

class GenerateStorkes():

    def __init__(self, painter_config, output_path):

        args = load_painter_config(painter_config)
        self.device = torch.device(f'cuda:{args.gpu_id}')
        self.pt = Painter(args=args)

        self.output_path = output_path

    @torch.no_grad()
    def generate_and_render(self, model, dataloader, ep):
        model.eval()
        data = next(iter(dataloader))
        data = dict_to_device(data, self.device, to_skip=['strokes', 'time_steps'])
        generated_strokes = model.generate(data)
        generated_strokes = generated_strokes.detach().cpu().numpy()

        # Check generated strokes
        checked_gen_strokes = self.pt.get_checked_strokes(generated_strokes)
        if len(checked_gen_strokes) == 0:
            print('Skipping because of wrong format strokes')
            np.save(os.path.join(self.output_path, f'wrong_strokes_ep_{ep}.npy'), generated_strokes)
            return {}
        else:
            original, _ = self.pt.inference(strokes=data['strokes_seq'].cpu().numpy())
            generated, _ = self.pt.inference(strokes=generated_strokes)
            plt.imsave(os.path.join(self.output_path, f'original_{ep}.jpg'), original)
            plt.imsave(os.path.join(self.output_path, f'generated_{ep}.jpg'), generated)

            return {'original' : original,
                    'generated' : generated}


