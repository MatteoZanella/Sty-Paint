from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np
import random
import os

"""
Dataset structure:
root_dir:
    - image_i:
        - strokes_params.npz 
        - index.pkl   # if we want to use different heuristics, maybe is better to reorder accordingly
        - image.jpg   # input image
        - render/
            - 0.png
            - 1.png
            .
            .
            .
            - j.png   # image rendered until the j-th, it is used to condition the transformer
"""

class StrokesDataset(Dataset):

    def __init__(self,
                 config):

        self.root_dir = config["dataset"]["root_dir"]
        self.filenames = sorted(os.listdir(self.root_dir))    # maybe check that every directory have the specified form before listing
        self.context_length = config["dataset"]["context_length"]
        self.sequence_length = config["dataset"]["sequence_length"]
        self.heuristic = config["dataset"]["heuristic"]
        self.img_size = config["dataset"]["resize"]

        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.filenames)

    def load_storkes(self, name):
        '''
        Format is 1 x T x 12   (x,y,h,w,theta,r1,g1,b1,r2,g2,b2,alpha)
        '''

        data = np.load(os.path.join(self.root_dir, name, 'strokes_params.npz'))
        strokes = np.concatenate([data['x_ctt'], data['x_color'], data['x_alpha']], axis=-1)
        strokes = torch.tensor(strokes, dtype=torch.float).squeeze(0)
        # TODO: Double is the default type, check if using float is good enoguh

        return strokes

    def sample_storkes(self, n):
        t = random.randint(self.context_length, n-self.sequence_length)
        t_C = t-self.context_length
        t_T = t+self.sequence_length
        return t_C, t, t_T

    def load_canvas_states(self, name, t_C, t_T):

        tmp_path = os.path.join(self.root_dir, name, 'render_' + self.heuristic)

        canvas = []
        for i in range(t_C, t_T):
            img = Image.open(os.path.join(tmp_path, f'{i}.jpg'))
            img = self.img_transform(img)
            canvas.append(img)

        canvas = torch.stack(canvas)

        return canvas

    def __getitem__(self, idx):
            name = self.filenames[idx]

            # ---------
            # Load Image
            img = Image.open(os.path.join(self.root_dir, name, name+'.jpg'))
            img = self.img_transform(img)

            # ---------
            # Load strokes and sample
            strokes = self.load_storkes(name)
            t_C, t, t_T = self.sample_storkes(strokes.shape[0])
            strokes = strokes[t_C:t_T, :]

            # ---------
            # Load rendered image up to s
            canvas_sequence = self.load_canvas_states(name, t_C, t_T)

            context = {'strokes' : strokes[:self.context_length, :],
                       'canvas' : canvas_sequence[:self.context_length, :, :, :]}

            x = {'strokes': strokes[self.context_length:, :],
                 'canvas' : canvas_sequence[self.context_length:, :, :, :]}


            return {'sequence' : x,
                    'context' : context,
                    'ref_img' : img}


class ToDevice:
    def __init__(self, device):
        self.device = device

    def move_dict_to(self, x : dict):
        out = {}

        for k, v in x.items():
            if isinstance(v, dict):
                out[k] = {}
                for k2, v2 in v.items():
                    out[k][k2] = v2.to(self.device)
            else:
                out[k] = v.to(self.device)

        return out


if __name__ == '__main__':

    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    from model.utils.parse_config import ConfigParser

    c_parser = ConfigParser('utils/config.yaml')
    c_parser.parse_config()
    config = c_parser.get_config()

    dataset = StrokesDataset(config=config)

    dataloader = DataLoader(dataset, batch_size=2)

    data = next(iter(dataloader))

    print(data['ref_img'].shape)