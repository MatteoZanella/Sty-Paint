import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()

        self.net = nn.Sequential(                                   # taken from https://arxiv.org/abs/2108.03798
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(3, 32, 3, 1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(32, 64, 3, 2),
                        nn.BatchNorm2d(64),
                        nn.ReLU(True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(64, 128, 3, 2),
                        nn.BatchNorm2d(128),
                        nn.AdaptiveAvgPool2d(1),
                        nn.ReLU(True))

    def forward(self, x):
        return self.net(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class ResNetEncoder(nn.Module):

    def __init__(self, type='resnet18'):
        super(ResNetEncoder, self).__init__()

        self.net = models.resnet18(pretrained=False)
        self.net.fc = Identity()

    def forward(self, x):
        x = self.net(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        """
        :param x: tensor of size bs x length x features to add positional emebddings
        """
        return x + self.pos_table[:, :x.size(1)].clone().detach()