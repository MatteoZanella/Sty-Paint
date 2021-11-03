from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np
import random
import os
import pickle
import pandas as pd

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

        self.active_sampling = config["dataset"]["sampling"]["active"]
        self.sampling_threshold = config["dataset"]["sampling"]["threshold"]
        self.sampling_p = config["dataset"]["sampling"]["prob"]


        self.df = pd.read_csv(self.config["dataset"]["csv_file"])
        self.root_dir = os.path.join(self.config["dataset"]["root"], partition, 'brushstrokes_generation_dataset')

        self.filenames = list(self.df[(self.df["partition"] == partition) & (self.df["isTrain"] == self.isTrain)]['filename'])

        # Configs
        self.context_length = config["dataset"]["context_length"]
        self.sequence_length = config["dataset"]["sequence_length"]
        self.heuristic = config["dataset"]["heuristic"]
        self.img_size = config["dataset"]["resize"]
        self.use_images = config["dataset"]["use_images"]

        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),])
            #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.filenames)

    def load_strokes(self, name):
        '''
        Format is 1 x T x 12   (x,y,h,w,theta,r1,g1,b1,r2,g2,b2,alpha)
        Exclude the alpha parameter: 1 x T x 11
        '''

        data = np.load(os.path.join(self.root_dir, name, 'strokes_params.npz'))
        color = 0.5*(data['x_color'][:, :, :3] + data['x_color'][:, :, 3:])
        strokes = np.concatenate([data['x_ctt'], color], axis=-1)
        strokes = torch.tensor(strokes, dtype=torch.float).squeeze(0)

        return strokes

    def sample_strokes(self, n):
        if self.active_sampling:
            if random.random() < self.sampling_p:
                n = self.sampling_threshold
        t = random.randint(self.context_length, n-self.sequence_length)
        t_C = t-self.context_length
        t_T = t+self.sequence_length

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
            'strokes_seq': strokes[self.context_length :, :]}
        # ---------
        if self.use_images:
            # Load rendered image up to s
            canvas = self.load_canvas_states(name, t-1)
            # ---------
            # Load Image
            img = Image.open(os.path.join(self.root_dir, name, name+'.jpg')).convert('RGB')
            img = self.img_transform(img)

            data.update({
                'canvas': canvas,
                'img': img})

        if not self.isTrain:
            data.update({'time_steps' : [t_C, t, t_T]})
            #data.update({'strokes' : all_strokes})  #TODO: fix here

        return data


########################################################################################################################
class EvalDataset(Dataset):

    def __init__(self,
                 config,
                 isTrain):

        self.config = config
        self.isTrain = isTrain

        # Load csv file
        partition = self.config["dataset"]["partition"]

        self.active_sampling = config["dataset"]["sampling"]["active"]
        self.sampling_threshold = config["dataset"]["sampling"]["threshold"]
        self.sampling_p = config["dataset"]["sampling"]["prob"]


        self.df = pd.read_csv(self.config["dataset"]["csv_file"])
        self.root_dir = os.path.join(self.config["dataset"]["root"], partition, 'brushstrokes_generation_dataset')

        self.filenames = list(self.df[(self.df["partition"] == partition) & (self.df["isTrain"] == self.isTrain)]['filename'])

        # Configs
        self.context_length = config["dataset"]["context_length"]
        self.sequence_length = config["dataset"]["sequence_length"]
        self.heuristic = config["dataset"]["heuristic"]
        self.img_size = config["dataset"]["resize"]
        self.use_images = config["dataset"]["use_images"]

        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),])
            #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.filenames)

    def load_strokes(self, name):
        '''
        Format is 1 x T x 12   (x,y,h,w,theta,r1,g1,b1,r2,g2,b2,alpha)
        Exclude the alpha parameter: 1 x T x 11
        '''

        data = np.load(os.path.join(self.root_dir, name, 'strokes_params.npz'))
        color = 0.5*(data['x_color'][:, :, :3] + data['x_color'][:, :, 3:])
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
            t = random.randint(self.context_length, n-self.sequence_length)
        t_C = t-self.context_length
        t_T = t+self.sequence_length

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
            'strokes_seq': strokes[self.context_length :, :]}
        # ---------
        if self.use_images:
            # Load rendered image up to s
            canvas = self.load_canvas_states(name, t-1)
            # ---------
            # Load Image
            img = Image.open(os.path.join(self.root_dir, name, name+'.jpg')).convert('RGB')
            img = self.img_transform(img)

            data.update({
                'canvas': canvas,
                'img': img})

        if not self.isTrain:
            data.update({'time_steps' : [t_C, t, t_T]})
            #data.update({'strokes' : all_strokes})  #TODO: fix here

        return data

    def sample(self, filename, timestep):
        # ---------
        # Load strokes, reorder and sample
        all_strokes = self.load_strokes(filename)
        idx = self.load_heuristic_idx(filename)
        all_strokes = all_strokes[idx]
        t_C, t, t_T = self.sample_strokes(all_strokes.shape[0], timestep)
        strokes = all_strokes[t_C :t_T, :]
        data = {
            'strokes_ctx' : strokes[:self.context_length, :][None],
            'strokes_seq' : strokes[self.context_length :, :][None]}
        # ---------
        if self.use_images :
            # Load rendered image up to s
            canvas = self.load_canvas_states(filename, t - 1)
            # ---------
            # Load Image
            img = Image.open(os.path.join(self.root_dir, filename, filename + '.jpg')).convert('RGB')
            img = self.img_transform(img)

            data.update({
                'canvas' : canvas[None],
                'img' : img[None]})

        if not self.isTrain :
            data.update({'time_steps' : [t_C, t, t_T]})
            # data.update({'strokes' : all_strokes})  #TODO: fix here

        return data