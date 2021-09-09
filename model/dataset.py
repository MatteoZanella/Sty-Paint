from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np
import random
import os
import pickle

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
                 config,
                 split):

        assert split == 'train' or split == 'test'
        self.split = split

        self.root_dir = config["dataset"][split]["root_dir"]
        self.filenames = sorted(os.listdir(self.root_dir))    # maybe check that every directory have the specified form before listing
        self.context_length = config["dataset"]["context_length"]
        self.sequence_length = config["dataset"]["sequence_length"]
        self.heuristic = config["dataset"]["heuristic"]
        self.img_size = config["dataset"]["resize"]
        self.debug = config["dataset"]["debug"]

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

        return strokes

    def sample_storkes(self, n, debug=False):
        if not debug:
            t = random.randint(self.context_length, n-self.sequence_length)
        else:
            t = 350  # fix it

        t_C = t-self.context_length
        t_T = t+self.sequence_length

        return t_C, t, t_T

    def load_heuristic_idx(self, name):
        file = os.path.join(self.root_dir, name, self.heuristic + '.pkl')
        with open(file, 'rb') as f:
            idx = pickle.load(f)
        return idx

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
            # Load strokes, reorder and sample
            all_strokes = self.load_storkes(name)
            idx = self.load_heuristic_idx(name)
            all_strokes = all_strokes[idx]
            t_C, t, t_T = self.sample_storkes(all_strokes.shape[0], debug=self.debug)
            strokes = all_strokes[t_C:t_T, :]
            # ---------
            # Load rendered image up to s
            canvas_sequence = self.load_canvas_states(name, t_C, t_T)

            data = {
                'strokes_ctx' : strokes[:self.context_length, :],
                'canvas_ctx' : canvas_sequence[:self.context_length, :, :, :],
                'strokes_seq' : strokes[self.context_length:, :],
                'canvas_seq' : canvas_sequence[self.context_length:, :, :, :],
                'img' : img
            }

            if self.split == 'test':
                data.update({'time_steps' : [t_C, t, t_T]})
                data.update({'strokes' : all_strokes})

            return data