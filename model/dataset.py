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

    def __init__(self,
                 config,
                 img_transform,
                 canvas_transform):

        """
        :param C: context to the sequence
        :param T: length of the sequence
        :param heuristic: type of heuristic to use (allow for multiple styles)
        """

        self.root_dir = config["dataset"]["root_dir"]
        self.filenames = sorted(os.listdir(self.root_dir))    # maybe check that every directory have the specified form before listing
        self.context_length = config["dataset"]["context_length"]
        self.sequence_length = config["dataset"]["sequence_length"]
        self.heuristic = config["dataset"]["heuristic"]
        self.img_transform = img_transform
        self.canvas_transform = canvas_transform

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

    def load_canvas_states(self, name, t_C, t, t_T):

        tmp_path = os.path.join(self.root_dir, name, 'render_' + self.heuristic)

        context = []
        for i in range(t_C, t):
            canvas = Image.open(os.path.join(tmp_path, f'{i}.jpg'))
            canvas = self.canvas_transform(canvas)
            context.append(canvas)

        context = torch.stack(context)

        sequence = []
        for i in range(t, t_T):
            canvas = Image.open(os.path.join(tmp_path, f'{i}.jpg'))
            canvas = self.canvas_transform(canvas)
            sequence.append(canvas)

        sequence = torch.stack(sequence)

        return context, sequence

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
            strokes_context = strokes[t_C:t, :]
            strokes_sequence = strokes[t:t_T, :]

            # ---------
            # Load rendered image up to s
            canvas_context, canvas_sequence = self.load_canvas_states(name, t_C, t, t_T)

            return img, strokes_context, strokes_sequence, canvas_context, canvas_sequence



if __name__ == '__main__':

    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    from parse_config import ConfigParser

    c_parser = ConfigParser('./config.yaml')
    c_parser.parse_config()
    config = c_parser.get_config()

    img_transform = transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor()])
    c_transform = transforms.Compose([transforms.ToTensor()])
    dataset = StrokesDataset(config=config,
                             img_transform=img_transform,
                             canvas_transform=c_transform)

    dataloader = DataLoader(dataset, batch_size=2)

    ref_imgs, strokes_ctx, strokes_seq, canvas_ctx, canvas_seq = next(iter(dataloader))

    print(ref_imgs.shape)
    print(strokes_ctx.shape)
    print(strokes_seq.shape)
    print(canvas_ctx.shape)
    print(canvas_seq.shape)