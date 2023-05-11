import argparse
import os
import sys

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.networks.efdm import load_pretrained_efdm


def get_args():
    # settings
    parser = argparse.ArgumentParser(description='STYLIZED NEURAL PAINTING')
    parser.add_argument('-d', '--dataset', default='../wikiart', help='WikiArt dataset folder location')
    parser.add_argument('-c', '--content_dataset', default='/data1/eperuzzo/INP_dataset/', help='Content dataset folder location')
    parser.add_argument('-o', '--output', default='../pretrained', help='Output weights folder location')   
    parser.add_argument('--decoder', default='../pretrained/efdm_decoder_iter_160000.pth', help='EFDM decoder weights')
    parser.add_argument('--vgg', default='../pretrained/vgg_normalised.pth', help='EFDM vgg weights')

    return parser.parse_args()

class StyleDataset(Dataset):
    def __init__(self, root_dir, filenames, transform=None):
        self.root_dir = root_dir
        self.filenames = list(filenames)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        style_path = os.path.join(self.root_dir, filename)
        style = self.transform(Image.open(style_path).convert('RGB'))
        return style

class ContentDataset(Dataset):
    def __init__(self, root_dir, df, transform=None, partition='oxford_pet_dataset'):
        self.root_dir = root_dir
        self.filenames = list(df[(df['partition'] == partition)]['filename'])
        self.transform = transform
        self.folder = os.path.join(root_dir, f'{partition}_v2', 'brushstrokes_generation_dataset')

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        content_path = os.path.join(self.folder, filename.split('.')[0], filename)
        style = self.transform(Image.open(content_path).convert('RGB'))
        return style

if __name__ == '__main__':

    args = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the transformations
    efdm_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    vgg_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mid_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Read the content csv
    csv_path = os.path.join(args.content_dataset, 'dataset_config.csv')
    cont_df = pd.read_csv(csv_path)
    content_ds = ContentDataset(args.content_dataset, cont_df, transform=vgg_transform)
    content_dl = DataLoader(content_ds, batch_size=32, shuffle=False, pin_memory=True, num_workers=5)

    # Read the Wikiart csv
    csv_path = os.path.join(args.dataset, 'wclasses_split.csv')
    df = pd.read_csv(csv_path)
    style_ds = StyleDataset(args.dataset, df['file'], transform=vgg_transform)
    indices = torch.randperm(len(style_ds))[:len(content_ds)]
    style_ds = Subset(style_ds, indices)
    style_dl = DataLoader(style_ds, batch_size=32, shuffle=False, pin_memory=True, num_workers=5)

    # VGG network
    vgg = models.vgg19(pretrained=True).features
    vgg.to(device).eval()
    
    # ==== Activations accumulation ====
    SIZE = len(content_ds)
    BATCH_SIZE = 16
    efdm = load_pretrained_efdm(args.vgg, args.decoder).to(device).eval()
    indices = torch.randperm(len(style_ds))[:SIZE]
    efdm_style_ds = Subset(StyleDataset(args.dataset, df['file'], transform=efdm_transform), indices)
    indices = torch.randperm(len(content_ds))[:SIZE]
    efdm_content_ds = Subset(ContentDataset(args.content_dataset, cont_df, transform=efdm_transform), indices)
    efdm_style_dl = DataLoader(efdm_style_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=5)
    efdm_content_dl = DataLoader(efdm_content_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=5)

    means = []
    for s_batch, c_batch in tqdm(zip(efdm_style_dl, efdm_content_dl), desc=f'Get the activations'):
        s_batch = s_batch.to(device, non_blocking=True)
        c_batch = c_batch.to(device, non_blocking=True)
        with torch.no_grad():
            output = efdm(c_batch, s_batch)
            x = mid_transform(output)
            i = 0
            for layer in vgg.children():
                x = layer(x)
                if isinstance(layer, nn.ReLU):
                    val = x.sum(0)
                    if i == len(means):
                        means.append(val)
                    else:
                        means[i] += val
                    i += 1
    means = [(x / SIZE).mean((1,2)) for x in means]
    
    # accum_ds = ConcatDataset([style_ds, content_ds])
    # accum_dl = DataLoader(accum_ds, batch_size=32, shuffle=False, pin_memory=True, num_workers=5)
    # means = []
    # for batch in tqdm(accum_dl, desc='Get the activations'):
    #     x = batch.to(device, non_blocking=True)
    #     i = 0
    #     with torch.no_grad():
    #         for layer in vgg.children():
    #             x = layer(x)
    #             if isinstance(layer, nn.ReLU):
    #                 val = x.sum(0)
    #                 if i == len(means):
    #                     means.append(val)
    #                 else:
    #                     means[i] += val
    #                 i += 1
    # means = [(x / len(accum_ds)).mean((1,2)) for x in means]
    
    # ==== Rescale the weights ====
    i = 0
    for layer in vgg.children():
        if isinstance(layer, nn.Conv2d):
            # weights layout is: (out_channels, in_channels, K_1, K_2)
            W, b = layer.weight.data, layer.bias.data
            if i > 0:
                # undo upstream normalization to restore scale of incoming channels
                W *= means[i - 1][None, :, None, None]
            # then normalize activations by rescaling both weights and biases
            b /= means[i]
            W /= means[i][:, None, None, None]
            layer.weight.data.copy_(W)
            layer.bias.data.copy_(b)
            i += 1
    
    # ==== Save the weights ====
    path = os.path.join(args.output, 'vgg_gatys_norm.pth')
    torch.save(vgg.state_dict(), path)

    # ==== Test the result ====
    # Test on content dataset
    means = []
    for batch in tqdm(content_dl, desc='Test on the content dataset'):
        x = batch.to(device, non_blocking=True)
        i = 0
        with torch.no_grad():
            for layer in vgg.children():
                x = layer(x)
                if isinstance(layer, nn.ReLU):
                    val = x.sum(0)
                    if i == len(means):
                        means.append(val)
                    else:
                        means[i] += val
                    i += 1
    means = [(x / len(content_ds)).mean((1,2)) for x in means]
    std = torch.stack([x.std() for x in means]).cpu()
    mu = torch.stack([x.mean() for x in means]).cpu()
    print('Std: ', std)
    print('Mu: ', mu)
    
    means = []
    for batch in tqdm(style_dl, desc='Test on the styles dataset'):
        x = batch.to(device, non_blocking=True)
        i = 0
        with torch.no_grad():
            for layer in vgg.children():
                x = layer(x)
                if isinstance(layer, nn.ReLU):
                    val = x.sum(0)
                    if i == len(means):
                        means.append(val)
                    else:
                        means[i] += val
                    i += 1
    means = [(x / len(style_ds)).mean((1,2)) for x in means]
    std = torch.stack([x.std() for x in means]).cpu()
    mu = torch.stack([x.mean() for x in means]).cpu()
    print('Std: ', std)
    print('Mu: ', mu)

    # Test on stylization dataset
    SIZE = 2048
    BATCH_SIZE = 16
    efdm = load_pretrained_efdm(args.vgg, args.decoder).to(device).eval()
    indices = torch.randperm(len(style_ds))[:SIZE]
    style_ds = Subset(StyleDataset(args.dataset, df['file'], transform=efdm_transform), indices)
    indices = torch.randperm(len(content_ds))[:SIZE]
    content_ds = Subset(ContentDataset(args.content_dataset, cont_df, transform=efdm_transform), indices)
    style_dl = DataLoader(style_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=5)
    content_dl = DataLoader(content_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=5)

    means = []
    for s_batch, c_batch in tqdm(zip(style_dl, content_dl), desc='Test on the stylized contents'):
        s_batch = s_batch.to(device, non_blocking=True)
        c_batch = c_batch.to(device, non_blocking=True)
        with torch.no_grad():
            output = efdm(c_batch, s_batch)
            x = mid_transform(output)
            i = 0
            for layer in vgg.children():
                x = layer(x)
                if isinstance(layer, nn.ReLU):
                    val = x.sum(0)
                    if i == len(means):
                        means.append(val)
                    else:
                        means[i] += val
                    i += 1
    means = [(x / SIZE).mean((1,2)) for x in means]
    std = torch.stack([x.std() for x in means]).cpu()
    mu = torch.stack([x.mean() for x in means]).cpu()
    print('Std: ', std)
    print('Mu: ', mu)
