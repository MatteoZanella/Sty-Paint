import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

# ----------------------------------------------------------------------------------------------------------------------

class SequenceMerger(nn.Module):

    def __init__(self, config):
        super(SequenceMerger, self).__init__()

        self.d_model = config["model"]["d_model"]
        self.s_params = config["model"]["n_strokes_params"]

        self.proj = nn.Linear(512 + self.s_params, self.d_model)


    def forward(self, strokes_params, canvas_feat, img_feat=None):

        x = torch.cat((canvas_feat, strokes_params), dim=-1)  # bs x L x d, concatenate along feature dim

        if img_feat is not None:
            bs = img_feat.size(0)
            img_feat = torch.cat((img_feat, torch.zeros(bs, self.s_params, device=img_feat.device)), dim=-1)   # pad image features
            img_feat = img_feat.unsqueeze(1) # length 1
            x = torch.cat((img_feat, x), dim=1)   # concatenate on length dimension

        # Project
        x = self.proj(x)
        return x

