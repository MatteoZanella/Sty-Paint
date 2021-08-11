from torch.utils.data import Dataset
from torchvision import transforms
import torch
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

    def __init__(self, root_dir, transform=transforms.ToTensor(), seq_length=100, heuristic=None):
        if not os.path.isdir(root_dir):
            raise Exception(f"{root_dir}' is not a dir")

        self.root_dir = root_dir
        self.dirs = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir)])    # maybe check that every directory have the specified form before listing
        self.seq_length = seq_length
        self.transform = transform

    def __len__(self):
        return len(self.dirs)

    def load_storkes(self, dir):
        '''
        Format is 1 x T x 12   (x,y,h,w,theta,r1,g1,b1,r2,g2,b2,alpha)
        '''

        data = np.load(os.path.join(self.root_dir, dir, 'strokes_params.npz'))
        strokes = np.concatenate([data['x_ctt'], data['x_color'], data['x_alpha']], axis=-1)

        return strokes

    def sample_storkes(self, n):
        start = random.randint(self.seq_length, n-self.seq_length)
        end = start + self.seq_length
        return start, end

    def __getitem__(self, idx):
        # ---------
        # Load Image
        img = Image.open(self.dirs[idx])

        # ---------
        # Load strokes and sample
        strokes = self.load_storkes(self.dirs[idx])
        s, e = self.sample_storkes(strokes.shape[1])

        # ---------
        # Load rendered image up to s
        canvas = Image.open(os.path.join(self.dirs[idx], 'render', '{}.jpg'.format(s)))

        # -------
        # Apply transformation
        img = self.transform(img)
        canvas = self.transform(canvas)
        strokes = torch.Tensor(strokes[:, s:e, :])

        return img, canvas, strokes