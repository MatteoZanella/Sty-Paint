import torch.nn as nn
import torchvision.models as models

# ----------------------------------------------------------------------------------------------------------------------
class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
        # taken from https://arxiv.org/abs/2108.03798
        self.net = nn.Sequential(
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
                        #nn.AdaptiveAvgPool2d(1),
                        nn.ReLU(True))

    def forward(self, x):
        return self.net(x)

# ----------------------------------------------------------------------------------------------------------------------

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class ResNetEncoder(nn.Module):

    def __init__(self, pretrained):
        super(ResNetEncoder, self).__init__()

        self.net = models.resnet18(pretrained=pretrained)
        self.net.fc = Identity()

    def forward(self, x):
        x = self.net(x)
        return x