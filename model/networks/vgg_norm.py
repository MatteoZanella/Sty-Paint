import torch
import torch.nn as nn

class VGG19Norm():
    _net = None
    
    @staticmethod
    def net(path):
        if VGG19Norm._net is None:
            VGG19Norm._net = VGG19Norm.fetch_net(path)
        return VGG19Norm._net

    @staticmethod
    def fetch_net(path):
        vgg = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  # relu1-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu2-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu2-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),  # relu3-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),  # relu4-1, this is the last layer used
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()  # relu5-4
        )
        vgg.load_state_dict(torch.load(path))
        vgg = vgg[:31].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        return vgg


# def load_normalised_vgg19(path):
#     vgg = nn.Sequential(
#         nn.Conv2d(3, 3, (1, 1)),
#         nn.ReflectionPad2d((1, 1, 1, 1)),
#         nn.Conv2d(3, 64, (3, 3)),
#         nn.ReLU(),  # relu1-1
#         nn.ReflectionPad2d((1, 1, 1, 1)),
#         nn.Conv2d(64, 64, (3, 3)),
#         nn.ReLU(),  # relu1-2
#         nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#         nn.ReflectionPad2d((1, 1, 1, 1)),
#         nn.Conv2d(64, 128, (3, 3)),
#         nn.ReLU(),  # relu2-1
#         nn.ReflectionPad2d((1, 1, 1, 1)),
#         nn.Conv2d(128, 128, (3, 3)),
#         nn.ReLU(),  # relu2-2
#         nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#         nn.ReflectionPad2d((1, 1, 1, 1)),
#         nn.Conv2d(128, 256, (3, 3)),
#         nn.ReLU(),  # relu3-1
#         nn.ReflectionPad2d((1, 1, 1, 1)),
#         nn.Conv2d(256, 256, (3, 3)),
#         nn.ReLU(),  # relu3-2
#         nn.ReflectionPad2d((1, 1, 1, 1)),
#         nn.Conv2d(256, 256, (3, 3)),
#         nn.ReLU(),  # relu3-3
#         nn.ReflectionPad2d((1, 1, 1, 1)),
#         nn.Conv2d(256, 256, (3, 3)),
#         nn.ReLU(),  # relu3-4
#         nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#         nn.ReflectionPad2d((1, 1, 1, 1)),
#         nn.Conv2d(256, 512, (3, 3)),
#         nn.ReLU(),  # relu4-1, this is the last layer used
#         nn.ReflectionPad2d((1, 1, 1, 1)),
#         nn.Conv2d(512, 512, (3, 3)),
#         nn.ReLU(),  # relu4-2
#         nn.ReflectionPad2d((1, 1, 1, 1)),
#         nn.Conv2d(512, 512, (3, 3)),
#         nn.ReLU(),  # relu4-3
#         nn.ReflectionPad2d((1, 1, 1, 1)),
#         nn.Conv2d(512, 512, (3, 3)),
#         nn.ReLU(),  # relu4-4
#         nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
#         nn.ReflectionPad2d((1, 1, 1, 1)),
#         nn.Conv2d(512, 512, (3, 3)),
#         nn.ReLU(),  # relu5-1
#         nn.ReflectionPad2d((1, 1, 1, 1)),
#         nn.Conv2d(512, 512, (3, 3)),
#         nn.ReLU(),  # relu5-2
#         nn.ReflectionPad2d((1, 1, 1, 1)),
#         nn.Conv2d(512, 512, (3, 3)),
#         nn.ReLU(),  # relu5-3
#         nn.ReflectionPad2d((1, 1, 1, 1)),
#         nn.Conv2d(512, 512, (3, 3)),
#         nn.ReLU()  # relu5-4
#     )
#     vgg.load_state_dict(torch.load(path))
#     return vgg[:31]