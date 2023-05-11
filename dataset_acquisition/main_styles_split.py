import argparse
import os
import random
import sys

import pandas as pd
import torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.networks.efdm import load_pretrained_efdm
from model.training.losses import VGG19StyleLossOriginal

def get_args():
    # settings
    parser = argparse.ArgumentParser(description='STYLIZED NEURAL PAINTING')
    parser.add_argument('-d', '--dataset', required=True, help='WikiArt dataset folder location')
    parser.add_argument('-c', '--content_dataset', required=True, help='Content dataset folder location')
    parser.add_argument('-t','--train_split', default=80., type=float, help='Percentage used for the training set')
    parser.add_argument('-N', '--size', default=0, type=int, help='Number of images per style to use. If zero, all images are used for the augmentation strategy')
    parser.add_argument('-q', '--quality', default=.12, type=float, help='Quality threshold')
    parser.add_argument('--decoder', default='../pretrained/efdm_decoder_iter_160000.pth', help='EFDM decoder weights')
    parser.add_argument('--vgg', default='../pretrained/vgg_normalised.pth', help='EFDM vgg weights')

    return parser.parse_args()

class StyleDataset(Dataset):
    def __init__(self, root_dir, filenames, transform=None, device='cuda:0'):
        self.root_dir = root_dir
        self.filenames = list(filenames)
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        style_path = os.path.join(self.root_dir, filename)
        style = self.transform(Image.open(style_path))
        return style

if __name__ == '__main__':

    args = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the models
    loss = VGG19StyleLossOriginal(content_layers = {'4_4'}, style_layers={'4_4', '5_4'}, content_weight=3e-5, style_weight=1).eval().to(device)
    efdm = load_pretrained_efdm(args.vgg, args.decoder).eval().to(device)

    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    vgg_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Read the Wikiart csv
    csv_path = os.path.join(args.dataset, 'wclasses.csv')
    df = pd.read_csv(csv_path)

    # Read the content csv
    NUM_CONTENTS = 25
    csv_path = os.path.join(args.content_dataset, 'dataset_config.csv')
    cont_df = pd.read_csv(csv_path)
    cont_folder = os.path.join(args.content_dataset, 'oxford_pet_dataset_v2', 'brushstrokes_generation_dataset')
    cont_files = cont_df[(cont_df['partition'] == 'oxford_pet_dataset')].sample(NUM_CONTENTS)['filename']
    cont_imgs = [Image.open(os.path.join(cont_folder, file.split('.')[0], file)) for file in cont_files]
    contents = torch.stack([transform(img) for img in cont_imgs]).to(device)
    vgg_contents = vgg_transform(contents)

    # Quality filtering
    dataset = StyleDataset(args.dataset, df['file'], transform=transform, device=device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=5)
    qualities = []
    for style in tqdm(dataloader, desc='Quality filtering'):
        # Do not change the contents size since batch size is 1
        style = style.repeat_interleave(len(contents), dim=0).to(device)
        vgg_style = vgg_transform(style)
        with torch.no_grad():
            outputs = efdm(contents, style)
            vgg_outputs = vgg_transform(outputs)
            quality = loss(vgg_outputs, vgg_contents, vgg_style).cpu()
        # Average the quality across the different contents
        qualities.append(quality.mean())
    qualities = torch.stack(qualities)
    # q_max = qualities.max()
    # q_min = qualities.min()
    # norm_quality = (qualities - q_min) / (q_max - q_min)

    quality_mask = (qualities > args.quality).numpy()
    print(f'Images above quality threshold: {quality_mask.sum()/len(quality_mask)*100:.2f}%')
    
    # Define the styles to use
    excluded_artists = set()
    excluded_genres = set()
    excluded_styles = set()
    styles = list(set(range(140, 167)) - excluded_styles)

    if args.size > 0:
        # Select N random images per style
        selected_idx = []
        genre_mask = ~df['genre'].isin(excluded_genres)
        artist_mask = ~df['artist'].isin(excluded_artists)
        clear_genres = True
        for style in styles:
            style_mask = df['style'] == style
            for _ in range(args.size):
                # Mask update
                mask = style_mask & genre_mask & artist_mask & quality_mask
                while not mask.any():
                    # Alternate between resetting artists and genres
                    if clear_genres:
                        genre_mask = ~df['genre'].isin(excluded_genres)
                    else:
                        artist_mask = ~df['artist'].isin(excluded_artists)
                    clear_genres = not clear_genres
                    # Update loosen mask
                    mask = style_mask & genre_mask & artist_mask & quality_mask
                # Extract row
                row = df[mask].sample()
                selected_idx.append(row.index.item())
                # Update masks
                used_artist = df['artist'] == row['artist'].item()
                artist_mask = artist_mask & ~used_artist
                used_genre = df['genre'] == row['genre'].item()
                genre_mask = genre_mask & ~used_genre
    else:
        style_mask = ~df['style'].isin(excluded_styles)
        genre_mask = ~df['genre'].isin(excluded_genres)
        artist_mask = ~df['artist'].isin(excluded_artists)
        selected_idx = df[style_mask & genre_mask & artist_mask & quality_mask].index
    
    df.loc[selected_idx, 'isTrain'] = False
    
    # Split the stiles into train and test sets
    
    # num_train_styles = round(len(styles) * args.train_split / 100)
    # train_styles = random.sample(styles, num_train_styles)
    # isTrain = df['style'].isin(train_styles)
    # df.loc[selected_idx, 'isTrain'] = isTrain

    selected_df = df.loc[selected_idx]
    for style in styles:
        style_df = selected_df[selected_df['style'] == style]
        num_samples = round(len(style_df) * args.train_split / 100)
        if num_samples > 0:
            train_idxs = style_df.sample(num_samples).index
            df.loc[train_idxs, 'isTrain'] = True



    # Save the csv
    df.loc[selected_idx].to_csv(os.path.join(args.dataset, 'wclasses_split.csv'), index=False)
