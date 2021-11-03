import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
import torchvision

class KLDivergence(nn.Module):

    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, mu, log_sigma):
        kl_loss = 1 + log_sigma - mu.pow(2) - log_sigma.exp()   # bs x n_features
        kl_loss = -0.5 * torch.sum(kl_loss, dim=-1)

        return torch.mean(kl_loss)


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, pred, labels):
        loss = (pred - labels).pow(2).sum(dim=[1, 2])   # sum over features and sequence length
        return loss.mean()


########################################################################################################################
class MSECalculator :

    def __init__(self, loss_weights) :
        self.MSE = nn.MSELoss(reduction='mean')

        self.lambda_position = loss_weights["position"]
        self.lambda_size = loss_weights["size"]
        self.lambda_theta = loss_weights["theta"]
        self.lambda_color = loss_weights["color"]
        self.lambda_color_img = loss_weights["color_img"]

        self.blur = torchvision.transforms.GaussianBlur(kernel_size=(7,7))
    def __call__(self,
                 predictions,
                 targets,
                 ref_imgs,
                 isEval=False) :
        '''
        Args:
            predictions: [bs x L x dim] predicted sequence
            targets: [bs x L x dim] target sequence
            ref_imgs: [bs x 3 x h x w] reference images

        Returns:
            loss
        '''
        # Slice prediction
        preds_position = predictions[:, :, :2]
        preds_size = predictions[:, :, 2:4]
        preds_theta = predictions[:, :, 4]
        preds_color = predictions[:, :, 5:]

        # Slice target
        target_position = targets[:, :, :2]
        target_size = targets[:, :, 2:4]
        target_theta = targets[:, :, 4]
        target_color = targets[:, :, 5:]

        # Compute losses
        mse_position = self.MSE(preds_position, target_position)
        mse_size = self.MSE(preds_size, target_size)
        mse_theta = self.MSE(preds_theta, target_theta)
        mse_color = self.MSE(preds_color, target_color)

        # Sum
        mse = mse_position * self.lambda_position + \
            mse_size * self.lambda_size + \
            mse_theta * self.lambda_theta + \
            mse_color * self.lambda_color

        # Record each loss separately
        losses = dict(mse_position=mse_position.item(),
                      mse_size=mse_size.item(),
                      mse_theta=mse_theta.item(),
                      mse_color=mse_color.item(),
                      mse=mse.item())

        # Sample color from reference images
        if self.lambda_color_img > 0 or isEval:
            # Extract labels from the reference image
            ref_imgs = self.blur(ref_imgs)
            ref_imgs = repeat(ref_imgs, 'bs ch h w -> (bs L) ch h w', L=predictions.size(1))
            grid = rearrange(preds_position, 'bs L dim -> (bs L) 1 1 dim')
            target_color_img = F.grid_sample(ref_imgs, 2 * grid - 1, align_corners=False)
            target_color_img = rearrange(target_color_img, '(bs L) ch 1 1 -> bs L ch', L=predictions.size(1))

            mse_color_img = self.MSE(preds_color, target_color_img)

            # Add to the main loss
            mse += mse_color_img * self.lambda_color_img
            losses.update({'mse_color_img' : mse_color_img.item()})

        return mse, losses