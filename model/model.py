import torch
import torch.nn as nn


class ImageEmbedder(nn.Module):

    def __init__(self):
        super().__init__()

        self.enc_img = nn.Sequential(    # taken from https://arxiv.org/abs/2108.03798
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
                        nn.AvgPool2d(kernel_size=8),
                        nn.ReLU(True))
        self.enc_canvas = nn.Sequential(    # taken from https://arxiv.org/abs/2108.03798
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
                        nn.AvgPool2d(kernel_size=8),
                        nn.ReLU(True),
                        )

        self.conv = nn.Conv2d(128 * 2, 512, 1)  # merge features form the two source images
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=2),
                                             num_layers=2)

    def forward(self, img, canvas):
        img_feat = self.enc_img(img)
        canvas_feat = self.enc_canvas(canvas)
        feat = torch.cat([img_feat, canvas_feat], dim=1)
        feat = self.conv(feat)
        feat = feat.flatten(2).permute(2, 0, 1).contiguous()    # Sequence x batch x n_features,   here we can add positional embeddings
        out = self.encoder(feat)
        return out

# ----------------------------------------------------------------------------------------------------------------------

class TransformerVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = 100
        self.n_params = 512

        self.encoder = nn.TransformerDecoder(decoder_layer=nn.TransformerDecoderLayer(d_model=self.n_params, nhead=2),
                                              num_layers=2)
        self.decoder = nn.TransformerDecoder(decoder_layer=nn.TransformerDecoderLayer(d_model=self.n_params, nhead=2),
                                              num_layers=2)


    def encode(self, x, context):
        bs = x.shape[1]

        # add learnable tokens
        mu = torch.randn(1, bs, self.n_params)
        log_var = torch.randn(1, bs, self.n_params)
        x = torch.cat([mu, log_var, x], dim=0)  # bs x (T+2) x d_model

        # Encode the input
        x = self.encoder(x, context)

        return x[0, :, :], x[1, :, :]

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return eps * std + mu

    def decode(self, context):
        inp = torch.randn(self.T, 1, self.n_params)    # create random tokens to be decoded
        out = self.decoder(inp, context.unsqueeze(0))
        return out

    def forward(self, params, context):
        params = params.permute(1, 0, 2).contiguous()  # length first
        mu, log_var = self.encode(params, context)
        z = self.reparametrize(mu, log_var)
        out = self.decode(z)   # z is the input, context comes from the other branch

        return out


# ----------------------------------------------------------------------------------------------------------------------

class Painter(nn.Module):

    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEmbedder()
        self.transformer_vae = TransformerVAE()

    def forward(self, img, canvas, params):
        ctx = self.image_encoder(img, canvas)

        pred = self.transformer_vae(params, ctx)

        return pred



if __name__ == '__main__':

    bs = 1
    T = 100 # number of strokes input to the model
    image = torch.randn([1, 3, 512, 512])
    canvas = torch.randn([1, 3, 512, 512])
    stroke_params = torch.randn([1, T, 512])


    net = Painter()
    pred = net(image, canvas, stroke_params)

    print(pred.shape)