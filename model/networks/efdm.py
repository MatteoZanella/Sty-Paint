import torch
import torch.nn as nn

from model.networks.vgg_norm import VGG19Norm


def exact_feature_distribution_matching(content_f, style_f):
    assert (content_f.size() == style_f.size())
    B, C, W, H = content_f.shape
    # Flatten the images features
    content_f = content_f.view(B, C, -1)
    style_f = style_f.view(B, C, -1)
    # Deep copy with sort
    _, index_content = torch.sort(content_f)
    value_style, _ = torch.sort(style_f)
    inverse_index = index_content.argsort(-1)
    new_content = content_f + (value_style.gather(-1, inverse_index) - content_f.detach())
    return new_content.view(B, C, W, H)


class EfdmStyleTransfer(nn.Module):
    def __init__(self, vgg, decoder):
        super().__init__()
        self.vgg = vgg.eval()
        self.decoder = decoder.eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, content, style, alpha=1.0):
        content_f = self.vgg(content)
        style_f = self.vgg(style)
        features = exact_feature_distribution_matching(content_f, style_f)
        features = features * alpha + content_f * (1 - alpha)
        output = self.decoder(features)
        return torch.clamp(output, 0, 1)

def load_pretrained_efdm(vgg_path, decoder_path):
    vgg = VGG19Norm.net(vgg_path)
    decoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)),
        )
    decoder.load_state_dict(torch.load(decoder_path))
    return EfdmStyleTransfer(vgg, decoder)
