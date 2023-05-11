import os
import random
import pickle

import PIL.Image as Image
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms

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
            - j.png   # image rendered until the j-th
"""


class StrokesDataset(Dataset):

    def __init__(self,
                 config,
                 isTrain):

        self.config = config
        self.isTrain = isTrain

        # Load csv file
        partition = self.config["dataset"]["partition"]

        self.df = pd.read_csv(self.config["dataset"]["csv_file"])
        self.root_dir = os.path.join(self.config["dataset"]["root"],
                                     partition + f'_{self.config["dataset"]["version"]}',
                                     'brushstrokes_generation_dataset')

        self.filenames = list(
            self.df[(self.df["partition"] == partition) & (self.df["isTrain"] == self.isTrain)]['filename'])

        # Configs
        self.context_length = config["dataset"]["context_length"]
        self.sequence_length = config["dataset"]["sequence_length"]
        self.heuristic = config["dataset"]["heuristic"]
        self.img_size = config["dataset"]["resize"]

        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(), ])
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.filenames)

    def load_strokes(self, name):
        '''
        Format is 1 x T x 11   (x,y,h,w,theta,r1,g1,b1,r2,g2,b2)
        Exclude the alpha parameter: 1 x T x 11
        '''

        data = np.load(os.path.join(self.root_dir, name, 'strokes_params.npz'))
        color = 0.5 * (data['x_color'][:, :, :3] + data['x_color'][:, :, 3:])
        strokes = np.concatenate([data['x_ctt'], color], axis=-1)
        strokes = torch.tensor(strokes, dtype=torch.float).squeeze(0)

        return strokes

    def sample_strokes(self, n):
        t = random.randint(self.context_length, n - self.sequence_length)
        t_C = t - self.context_length
        t_T = t + self.sequence_length

        return t_C, t, t_T

    def load_heuristic_idx(self, name):
        file = os.path.join(self.root_dir, name, self.heuristic + '.pkl')
        with open(file, 'rb') as f:
            idx = pickle.load(f)
        return idx

    def load_canvas_states(self, name, time_step):
        tmp_path = os.path.join(self.root_dir, name, 'render_' + self.heuristic)
        canvas = Image.open(os.path.join(tmp_path, f'{str(time_step)}.jpg'))
        canvas = self.img_transform(canvas)
        return canvas

    def __getitem__(self, idx):
        name = self.filenames[idx]
        name = name.split('.')[0]
        # ---------
        # Load strokes, reorder and sample
        all_strokes = self.load_strokes(name)
        idx = self.load_heuristic_idx(name)
        all_strokes = all_strokes[idx]
        t_C, t, t_T = self.sample_strokes(all_strokes.shape[0])
        strokes = all_strokes[t_C:t_T, :]
        data = {
            'strokes_ctx': strokes[:self.context_length, :],
            'strokes_seq': strokes[self.context_length:, :]}
        # ---------
        # Load rendered image up to s
        canvas = self.load_canvas_states(name, t - 1)
        # ---------
        # Load Image
        img = Image.open(os.path.join(self.root_dir, name, name + '.jpg')).convert('RGB')
        img = self.img_transform(img)

        data.update({
            'canvas': canvas,
            'img': img})

        if not self.isTrain:
            data.update({'time_steps': torch.tensor([t_C, t, t_T])})
            # data.update({'strokes' : all_strokes})  #TODO: fix here

        return data


class StylizedStrokesDataset(Dataset):

    def __init__(self,
                 config,
                 isTrain):

        self.config = config
        self.isTrain = isTrain
        # ==== Content images ====
        # Load csv file
        partition = self.config["dataset"]["partition"]

        self.df = pd.read_csv(self.config["dataset"]["csv_file"])
        self.root_dir = os.path.join(self.config["dataset"]["root"],
                                     partition + f'_{self.config["dataset"]["version"]}',
                                     'brushstrokes_generation_dataset')
        mask = (self.df["partition"] == partition) & (self.df["isTrain"] == self.isTrain)
        self.filenames = self.df[mask]['filename'].tolist()
        # ==== Style images ====
        self.style_df = pd.read_csv(self.config["dataset"]["styles"]["csv_file"])
        self.style_root_dir = self.config["dataset"]["styles"]["root"]
        mask = self.style_df['isTrain'] == self.isTrain
        self.style_filenames = self.style_df[mask]['file'].tolist()
        # ==== Configs ====
        self.context_length = config["dataset"]["context_length"]
        self.sequence_length = config["dataset"]["sequence_length"]
        self.heuristic = config["dataset"]["heuristic"]
        self.img_size = config["stylization"]["resize"]
        self.augment = config["stylization"]["augment"]

        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(), ])

    def load_strokes(self, name):
        '''
        Format is 1 x T x 11   (x,y,h,w,theta,r1,g1,b1,r2,g2,b2)
        Exclude the alpha parameter: 1 x T x 11
        '''
        data = np.load(os.path.join(self.root_dir, name, 'strokes_params.npz'))
        color = 0.5 * (data['x_color'][:, :, :3] + data['x_color'][:, :, 3:])
        strokes = np.concatenate([data['x_ctt'], color], axis=-1)
        strokes = torch.tensor(strokes, dtype=torch.float).squeeze(0)
        return strokes

    def sample_strokes(self, n):
        t = random.randint(self.context_length, n - self.sequence_length)
        t_C = t - self.context_length
        t_T = t + self.sequence_length
        return t_C, t, t_T

    def load_heuristic_idx(self, name):
        file = os.path.join(self.root_dir, name, self.heuristic + '.pkl')
        with open(file, 'rb') as f:
            idx = pickle.load(f)
        return idx

    def __len__(self):
        return len(self.filenames) if self.augment else len(self.filenames) * len(self.style_filenames)

    def __getitem__(self, idx):
        if self.augment:
            content_idx = idx
            style_idx = random.randrange(len(self.style_filenames))
        else:
            content_idx = idx % len(self.filenames)
            style_idx = idx // len(self.filenames)
        content_name = self.filenames[content_idx]
        content_name = content_name.split('.')[0]
        # ---------
        # Load strokes, reorder and sample
        all_strokes = self.load_strokes(content_name)
        idx = self.load_heuristic_idx(content_name)
        all_strokes = all_strokes[idx]
        t_C, t, t_T = self.sample_strokes(all_strokes.shape[0])
        # We are not interested in the strokes after t_T
        strokes = all_strokes[:t_T, :]
        # ---------
        # Load content Image
        content = Image.open(os.path.join(self.root_dir, content_name, content_name + '.jpg')).convert('RGB')
        content = self.img_transform(content)
        # Load style Image
        style_filename = self.style_filenames[style_idx]
        style = Image.open(os.path.join(self.style_root_dir, style_filename)).convert('RGB')
        style = self.img_transform(style)
        # ---------
        data = {
            'time_steps': torch.tensor([t_C, t, t_T]),
            'strokes': strokes,
            'content': content,
            'style': style}

        return data

# ======================================================================================================================
class EvalDataset(Dataset):

    def __init__(self,
                 config,
                 isTrain):

        self.config = config
        self.isTrain = isTrain

        # Load csv file
        partition = self.config["dataset"]["partition"]

        self.df = pd.read_csv(self.config["dataset"]["csv_file"])
        self.root_dir = os.path.join(self.config["dataset"]["root"],
                                     partition + f'_{self.config["dataset"]["version"]}',
                                     'brushstrokes_generation_dataset')

        self.filenames = list(
            self.df[(self.df["partition"] == partition) & (self.df["isTrain"] == self.isTrain)]['filename'])

        # Configs
        self.context_length = config["dataset"]["context_length"]
        self.sequence_length = config["dataset"]["sequence_length"]
        self.heuristic = config["dataset"]["heuristic"]
        self.img_size = config["dataset"]["resize"]

        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(), ])
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.filenames)

    def load_strokes(self, name):
        '''
        Format is 1 x T x 12   (x,y,h,w,theta,r1,g1,b1,r2,g2,b2,alpha)
        Exclude the alpha parameter: 1 x T x 11
        '''

        data = np.load(os.path.join(self.root_dir, name, 'strokes_params.npz'))
        color = 0.5 * (data['x_color'][:, :, :3] + data['x_color'][:, :, 3:])
        strokes = np.concatenate([data['x_ctt'], color], axis=-1)
        strokes = torch.tensor(strokes, dtype=torch.float).squeeze(0)

        return strokes

    def sample_strokes(self, n, ts=None):
        if self.active_sampling:
            if random.random() < self.sampling_p:
                n = self.sampling_threshold
        if ts is not None:
            t = ts
        else:
            t = random.randint(self.context_length, n - self.sequence_length)
        t_C = t - self.context_length
        t_T = t + self.sequence_length

        return t_C, t, t_T

    def load_heuristic_idx(self, name):
        file = os.path.join(self.root_dir, name, self.heuristic + '.pkl')
        with open(file, 'rb') as f:
            idx = pickle.load(f)
        return idx

    def load_canvas_states(self, name, time_step):
        tmp_path = os.path.join(self.root_dir, name, 'render_' + self.heuristic)
        canvas = Image.open(os.path.join(tmp_path, f'{str(time_step)}.jpg'))
        canvas = self.img_transform(canvas)
        return canvas

    def __getitem__(self, idx):
        name = self.filenames[idx]
        name = name.split('.')[0]
        # ---------
        # Load strokes, reorder and sample
        all_strokes = self.load_strokes(name)
        idx = self.load_heuristic_idx(name)
        all_strokes = all_strokes[idx]
        t_C, t, t_T = self.sample_strokes(all_strokes.shape[0])
        strokes = all_strokes[t_C:t_T, :]
        data = {
            'strokes_ctx': strokes[:self.context_length, :],  # context strokes
            'strokes_seq': strokes[self.context_length:, :]}  # ground truth strokes
        # ---------
        # Load rendered image up to s
        canvas = self.load_canvas_states(name, t - 1)
        final_canvas = self.load_canvas_states(name, t_T)
        # ---------
        # Load Image
        img = Image.open(os.path.join(self.root_dir, name, name + '.jpg')).convert('RGB')
        img = self.img_transform(img)

        data.update({
            'canvas': canvas,
            'final_canvas': final_canvas,
            'img': img})

        if not self.isTrain:
            data.update({'time_steps': [t_C, t, t_T]})
            # data.update({'strokes' : all_strokes})  #TODO: fix here

        return data

    def sample(self, filename, timestep, tot=8):
        # ---------
        # Load strokes, reorder and sample
        all_strokes = self.load_strokes(filename)
        idx = self.load_heuristic_idx(filename)
        all_strokes = all_strokes[idx]
        t_C, t, t_T = self.sample_strokes(all_strokes.shape[0], timestep)

        initial_context = all_strokes[:t_C]
        original_sequence = all_strokes[t:t + tot]

        strokes = all_strokes[t_C:t_T, :]
        data = {
            'strokes_ctx': strokes[:self.context_length, :][None],
            'strokes_seq': strokes[self.context_length:, :][None]}
        # ---------
        # Load rendered image up to s
        canvas = self.load_canvas_states(filename, t - 1)
        # ---------
        # Load Image
        img = Image.open(os.path.join(self.root_dir, filename, filename + '.jpg')).convert('RGB')
        img = self.img_transform(img)

        data.update({
            'canvas': canvas[None],
            'img': img[None]})

        return data, initial_context, original_sequence[None]

