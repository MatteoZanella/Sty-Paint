from typing import Any, Dict, List

import torch
import torchvision.transforms as transforms
from torch.utils.data._utils.collate import default_collate

from model.utils.utils import dict_to_device
from model.networks.efdm import load_pretrained_efdm
from model.networks.light_renderer import LightRenderer
from model.style_transfer.stylization import stylize_strokes


def collate_strokes(batch, key : str ='strokes'):
    """	
    Collate function for strokes. Uses the default collate function, except for the 'key'.
    The values in the 'key' are concatenated instead of stacked in a new batch dimension.
    """
    strokes = []
    for sample in batch:
        strokes.append(sample.pop(key))
    collate = default_collate(batch)
    collate[key] = torch.cat(strokes)
    return collate

@torch.jit.script
def _slice_strokes(strokes, lengths : List[int], starts, ends):
    groups = strokes.split(lengths)
    return torch.cat([g[s:e] for g, s, e in zip(groups, starts, ends)])


def slice_strokes(strokes, lengths : List[int], starts = None, ends = None):
    """
    Slices stacked strokes from different samples, according to the given starts and ends.
    Args:
        strokes: Tensor of shape [L, ...].
        lengths: List of length [bs]. Each entry is the number of strokes for the corresponding sample.
        starts: The indexes of the first stroke to slice from each sample group. Can be a tensor of shape [bs],
            or a integer scalar if the index is the same for every group. If None, it slices from the beginning.
        ends: The indexes of the last stroke to slice from each sample group. Can be a tensor of shape [bs],
            or a integer scalar if the index is the same for every group. If None, it slices to the end.
    Returns:
        Tensor of shape [K, ...].
    """

    # If no start and end are provided, use the entire strokes
    starts = torch.zeros((len(lengths),), dtype=int) if starts is None else starts
    ends = torch.tensor(lengths) if ends is None else ends
    # If starts and ends are single values, convert them to tensors
    starts = torch.full((len(lengths),), starts) if isinstance(starts, int) else starts
    ends = torch.full((len(lengths),), ends) if isinstance(ends, int) else ends
    
    return _slice_strokes(strokes, lengths, starts, ends)


class DataLoaderWrapper:
    """
    Moves the batch to the correct device. If the stylization is enabled, the batch is also stylized.
    """
    def __init__(self, dl, efdm_model=None):
        self.dl = dl
        config = self.dl.dataset.config
        self.device = config['device']
        self.stylization = config["stylization"]["apply"]
        self.ctx_len = config['dataset']['context_length']
        self.seq_len = config['dataset']['sequence_length']
        self.stylize_img = "stylize_img" not in config["stylization"] or config["stylization"]["stylize_img"]
        self.stylize_strokes_ctx = "stylize_strokes_ctx" not in config["stylization"] or config["stylization"]["stylize_strokes_ctx"]

        # Stylization config
        if "model" in config and "train" in config:
            use_style = "use_style" in config["model"]["context_encoder"] and config["model"]["context_encoder"]["use_style"]
            use_style_loss = "vgg_style" in config["train"]["losses"] and config["train"]["losses"]["vgg_style"]["weight"] > 0.0
            self.batch_with_style = use_style or use_style_loss
        else:
            self.batch_with_style = True

        img_size = config['dataset']['resize']
        self.img_transform = transforms.Resize((img_size, img_size))

        if self.stylization:
            # EFDM Stylizer
            if efdm_model is None:
                vgg_path = config["stylization"]["vgg_weights"]
                decoder_path = config["stylization"]["decoder_weights"]
                self.efdm = load_pretrained_efdm(vgg_path, decoder_path).eval().to(self.device)
            else:
                self.efdm = efdm_model
            # Light renderer
            brush_paths = config["stylization"]["brush_paths"]
            batch_size = config["stylization"]["renderer_batch_size"]
            self.renderer = LightRenderer(brush_paths, img_size, batch_size=batch_size).to(self.device)

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in iter(self.dl):
            b = dict_to_device(b, device=self.device)
            if self.stylization:
                b = self.style_preprocess(b)
            yield b
    
    def style_preprocess(self, batch):
        # Lengths of the batches : t_T
        t_T = batch['time_steps'][:, 2].tolist()
        # Stylize the content image with EFDM
        with torch.no_grad():
            sty_img = self.efdm(batch['content'], batch['style'])
        sty_img = self.img_transform(sty_img)
        # Stylize the strokes with the stylized image
        strokes_t_T = stylize_strokes(sty_img, batch['strokes'], t_T, rescaling=.33)
        # Slice the strokes [:t]
        t = batch['time_steps'][:, 1]

        strokes_t = slice_strokes(strokes_t_T, t_T, ends=t)
        # strokes_t = slice_strokes(batch['strokes'], t_T, ends=t)

        # Render the canvas with the strokes up to t step, the new lengths
        canvases = self.renderer(strokes_t, t.tolist())
        # Extract the context + sequence from the stylized strokes
        tot_length = self.ctx_len + self.seq_len
        if not self.stylize_strokes_ctx:
            # Use the original strokes for the prediction
            strokes_ori = slice_strokes( batch['strokes'], t_T, starts=-tot_length)
            strokes_ori = strokes_ori.unflatten(0, (len(t), tot_length))
        strokes_sty = slice_strokes(strokes_t_T, t_T, starts=-tot_length)
        strokes_sty = strokes_sty.unflatten(0, (len(t), tot_length))
        # Resize the style
        style = self.img_transform(batch['style'])

        style_batch = {
            'canvas' : canvases,
            'strokes_ctx': strokes_sty[:, :self.ctx_len] if self.stylize_strokes_ctx else strokes_ori[:, :self.ctx_len],
            'strokes_seq': strokes_sty[:, self.ctx_len:],
        }

        if self.stylize_img:
            style_batch['img'] = sty_img
        else:
            style_batch['img'] = self.img_transform(batch['content'])
            style_batch['img_target'] = sty_img

        if self.batch_with_style:
            style_batch['style'] = style
        
        del batch

        return style_batch