from typing import List

import torch

from model.style_transfer.approximation import approximate_strokes, sample_windows


@torch.jit.script
def extract_colors(samples: List[torch.Tensor]):
    """
    Take mean color for each sample in the list.
    Args:
        samples: List containing [L] tensors of varying shape [3, hr, wr]
    """
    return torch.stack([sample.mean(dim=(1, 2)) for sample in samples])


@torch.jit.script
def stylize_strokes(stylized, strokes, lengths : List[int], rescaling : float = .75):
    windows = approximate_strokes(strokes, rescaling=rescaling)
    samples = sample_windows(stylized, windows, lengths)
    colors = extract_colors(samples)
    # With more extractions, the rescaling should be moved at color level

    return torch.cat((strokes[..., :-3], colors), dim=-1)