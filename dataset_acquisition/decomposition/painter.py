import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

from . import morphology, loss, utils, renderer
from .networks import *

import torch
import torch.optim as optim

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PainterBase():
    def __init__(self, args):
        self.args = args
        self.rderr = renderer.Renderer(brush_paths=args.brush_paths,
                                       renderer=args.renderer,
                                       CANVAS_WIDTH=args.canvas_size,
                                       canvas_color=args.canvas_color,
                                       )

        # define G
        self.device = torch.device(f'cuda:{args.gpu_id}')
        self.net_G = define_G(rdrr=self.rderr, netG=args.net_G,device=self.device).to(self.device)

        # define some other vars to record the training states
        self.x_ctt = None
        self.x_color = None
        self.x_alpha = None

        self.G_pred_foreground = None
        self.G_pred_alpha = None
        self.G_final_pred_canvas = torch.zeros(
            [1, 3, self.net_G.out_size, self.net_G.out_size]).to(self.device)

        self.G_loss = torch.tensor(0.0)
        self.step_id = 0
        self.anchor_id = 0
        self.renderer_checkpoint_dir = args.renderer_checkpoint_dir
        self.output_dir = args.output_dir
        self.lr = args.lr

        # define the loss functions
        self._pxl_loss = loss.PixelLoss(p=1)
        self._sinkhorn_loss = loss.SinkhornLoss(epsilon=0.01, niter=5, normalize=False)

        # some other vars to be initialized in child classes
        self.input_aspect_ratio = None
        self.img_path = None
        self.img_batch = None
        self.img_ = None
        self.final_rendered_images = None
        self.m_grid = None
        self.m_strokes_per_block = None

    def _load_checkpoint(self):

        # load renderer G
        if os.path.exists((os.path.join(
                self.renderer_checkpoint_dir, 'last_ckpt.pt'))):
            print('loading renderer from pre-trained checkpoint...')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.renderer_checkpoint_dir, 'last_ckpt.pt'),
                                map_location=self.device)
            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            self.net_G.to(self.device)
            self.net_G.eval()
        else:
            print('pre-trained renderer does not exist...')
            exit()


    def _compute_acc(self):

        target = self.img_batch.detach()
        canvas = self.G_pred_canvas.detach()
        psnr = utils.cpt_batch_psnr(canvas, target, PIXEL_MAX=1.0)

        return psnr

    def _save_stroke_params(self, v):

        d_shape = self.rderr.d_shape
        d_color = self.rderr.d_color
        d_alpha = self.rderr.d_alpha

        x_ctt = v[:, :, 0:d_shape]
        x_color = v[:, :, d_shape:d_shape+d_color]
        x_alpha = v[:, :, d_shape+d_color:d_shape+d_color+d_alpha]
        print('saving stroke parameters...')
        np.savez(os.path.join(self.output_dir, 'strokes_params.npz'), x_ctt=x_ctt,
                 x_color=x_color, x_alpha=x_alpha)

    def _shuffle_strokes_and_reshape(self, v):

        grid_idx = list(range(self.m_grid ** 2))
        random.shuffle(grid_idx)
        v = v[grid_idx, :, :]
        v = np.reshape(np.transpose(v, [1,0,2]), [-1, self.rderr.d])
        v = np.expand_dims(v, axis=0)

        return v

    def _render(self, v, save_jpgs=True, save_video=True):

        v = v[0,:,:]
        if self.args.keep_aspect_ratio:
            if self.input_aspect_ratio < 1:
                out_h = int(self.args.canvas_size * self.input_aspect_ratio)
                out_w = self.args.canvas_size
            else:
                out_h = self.args.canvas_size
                out_w = int(self.args.canvas_size / self.input_aspect_ratio)
        else:
            out_h = self.args.canvas_size
            out_w = self.args.canvas_size

        file_name = os.path.join(
            self.output_dir, self.img_path.split('/')[-1][:-4])

        if save_video:
            video_writer = cv2.VideoWriter(
                file_name + '_animated.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 40,
                (out_w, out_h))

        print('rendering canvas...')
        self.rderr.create_empty_canvas()
        for i in range(v.shape[0]):  # for each stroke
            self.rderr.stroke_params = v[i, :]
            if self.rderr.check_stroke():
                self.rderr.draw_stroke()
            this_frame = self.rderr.canvas
            this_frame = cv2.resize(this_frame, (out_w, out_h), cv2.INTER_AREA)
            if save_jpgs:
                plt.imsave(file_name + '_rendered_stroke_' + str((i+1)).zfill(4) +
                           '.png', this_frame)
            if save_video:
                video_writer.write((this_frame[:,:,::-1] * 255.).astype(np.uint8))

        if save_jpgs:
            print('saving input photo...')
            out_img = cv2.resize(self.img_, (out_w, out_h), cv2.INTER_AREA)
            plt.imsave(file_name + '_input.png', out_img)

        final_rendered_image = np.copy(this_frame)
        if save_jpgs:
            print('saving final rendered result...')
            plt.imsave(file_name + '_final.png', final_rendered_image)

        return final_rendered_image




    def _normalize_strokes(self, v):

        v = np.array(v.detach().cpu())

        if self.rderr.renderer in ['watercolor', 'markerpen']:
            # x0, y0, x1, y1, x2, y2, radius0, radius2, ...
            xs = np.array([0, 4])
            ys = np.array([1, 5])
            rs = np.array([6, 7])
        elif self.rderr.renderer in ['oilpaintbrush', 'rectangle']:
            # xc, yc, w, h, theta ...
            xs = np.array([0])
            ys = np.array([1])
            rs = np.array([2, 3])
        else:
            raise NotImplementedError('renderer [%s] is not implemented' % self.rderr.renderer)

        for y_id in range(self.m_grid):
            for x_id in range(self.m_grid):
                y_bias = y_id / self.m_grid
                x_bias = x_id / self.m_grid
                v[y_id * self.m_grid + x_id, :, ys] = \
                    y_bias + v[y_id * self.m_grid + x_id, :, ys] / self.m_grid
                v[y_id * self.m_grid + x_id, :, xs] = \
                    x_bias + v[y_id * self.m_grid + x_id, :, xs] / self.m_grid
                v[y_id * self.m_grid + x_id, :, rs] /= self.m_grid

        return v


    def initialize_params(self):

        self.x_ctt = np.random.rand(
            self.m_grid*self.m_grid, self.m_strokes_per_block,
            self.rderr.d_shape).astype(np.float32)
        self.x_ctt = torch.tensor(self.x_ctt).to(self.device)

        self.x_color = np.random.rand(
            self.m_grid*self.m_grid, self.m_strokes_per_block,
            self.rderr.d_color).astype(np.float32)
        self.x_color = torch.tensor(self.x_color).to(self.device)

        self.x_alpha = np.random.rand(
            self.m_grid*self.m_grid, self.m_strokes_per_block,
            self.rderr.d_alpha).astype(np.float32)
        self.x_alpha = torch.tensor(self.x_alpha).to(self.device)


    def stroke_sampler(self, anchor_id):

        if anchor_id == self.m_strokes_per_block:
            return

        err_maps = torch.sum(
            torch.abs(self.img_batch - self.G_final_pred_canvas),
            dim=1, keepdim=True).detach()

        for i in range(self.m_grid*self.m_grid):
            this_err_map = err_maps[i,0,:,:].cpu().numpy()
            ks = int(this_err_map.shape[0] / 8)
            this_err_map = cv2.blur(this_err_map, (ks, ks))
            this_err_map = this_err_map ** 4
            this_img = self.img_batch[i, :, :, :].detach().permute([1, 2, 0]).cpu().numpy()

            self.rderr.random_stroke_params_sampler(
                err_map=this_err_map, img=this_img)

            self.x_ctt.data[i, anchor_id, :] = torch.tensor(
                self.rderr.stroke_params[0:self.rderr.d_shape])
            self.x_color.data[i, anchor_id, :] = torch.tensor(
                self.rderr.stroke_params[self.rderr.d_shape:self.rderr.d_shape+self.rderr.d_color])
            self.x_alpha.data[i, anchor_id, :] = torch.tensor(self.rderr.stroke_params[-1])


    def _backward_x(self):

        self.G_loss = 0
        self.G_loss += self.args.beta_L1 * self._pxl_loss(
            canvas=self.G_final_pred_canvas, gt=self.img_batch)
        if self.args.with_ot_loss:
            self.G_loss += self.args.beta_ot * self._sinkhorn_loss(
                self.G_final_pred_canvas, self.img_batch)
        self.G_loss.backward()


    def _forward_pass(self):

        self.x = torch.cat([self.x_ctt, self.x_color, self.x_alpha], dim=-1)

        v = torch.reshape(self.x[:, 0:self.anchor_id+1, :],
                          [self.m_grid*self.m_grid*(self.anchor_id+1), -1, 1, 1])
        self.G_pred_foregrounds, self.G_pred_alphas = self.net_G(v)

        self.G_pred_foregrounds = morphology.Dilation2d(m=1)(self.G_pred_foregrounds)
        self.G_pred_alphas = morphology.Erosion2d(m=1)(self.G_pred_alphas)

        self.G_pred_foregrounds = torch.reshape(
            self.G_pred_foregrounds, [self.m_grid*self.m_grid, self.anchor_id+1, 3,
                                      self.net_G.out_size, self.net_G.out_size])
        self.G_pred_alphas = torch.reshape(
            self.G_pred_alphas, [self.m_grid*self.m_grid, self.anchor_id+1, 3,
                                 self.net_G.out_size, self.net_G.out_size])

        for i in range(self.anchor_id+1):
            G_pred_foreground = self.G_pred_foregrounds[:, i]
            G_pred_alpha = self.G_pred_alphas[:, i]
            self.G_pred_canvas = G_pred_foreground * G_pred_alpha \
                                 + self.G_pred_canvas * (1 - G_pred_alpha)

        self.G_final_pred_canvas = self.G_pred_canvas


########################################################################################################################
# Modify this class
class Painter(PainterBase):

    def __init__(self, args):
        super(Painter, self).__init__(args=args)
        self.args = args

        self._load_checkpoint()
        self.net_G.eval()
        print(f'Painter created, weights form: {args.renderer_checkpoint_dir}, eval mode: True')

    def load_style_image(self):
        style_img = cv2.imread(self.img_path, cv2.IMREAD_COLOR)
        self.style_img_ = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        self.style_img = cv2.blur(cv2.resize(self.style_img_, (128, 128)), (2, 2))
        self.style_img = torch.tensor(self.style_img).permute([2, 0, 1]).unsqueeze(0).to(device)

    def _backward_x(self):

        self.G_loss = 0
        self.G_loss += self.args.beta_L1 * self._pxl_loss(
            canvas=self.G_final_pred_canvas, gt=self.img_batch)
        if self.args.with_ot_loss:
            self.G_loss += self.args.beta_ot * self._sinkhorn_loss(
                self.G_final_pred_canvas, self.img_batch)

        # with torch.no_grad():
        #     canvas = utils.patches2img(
        #         self.G_final_pred_canvas, self.m_grid, to_numpy=False).to(device)
        #     style_img = utils.patches2img(self.img_batch, self.m_grid, to_numpy=False).to(device)
        #
        #     assert canvas.shape == style_img.shape
        #     content_loss = self._content_loss(canvas, style_img)
        #     style_loss = self._style_loss(canvas, style_img)
        #
        #     """
        #     style_loss = 0.0
        #     content_loss = 0.0
        #     N = self.G_final_pred_canvas.shape[0]
        #     for i in range(N):
        #         gi = self.G_final_pred_canvas[i].unsqueeze(0)
        #         ii = self.img_batch[i].unsqueeze(0)
        #         style_loss += self._style_loss(gi, ii)
        #         content_loss += self._content_loss(gi, ii)
        #     style_loss /= N
        #     content_loss /= N
        #     #style_loss = self._style_loss(canvas, self.style_img)
        #     #content_loss = self._content_loss(canvas, self.style_img)
        #     """
        # self.loss_dict['pixel_loss'].append(self.G_loss.item())
        # self.loss_dict['style_loss'].append(style_loss.item())
        # self.loss_dict['content_loss'].append(content_loss.item())

        self.G_loss.backward()


    def manual_set_number_strokes_per_block(self, id):
        self.m_strokes_per_block = self.manual_strokes_per_block[id]

    def stroke_parser(self):

        total_blocks = 0
        for i in range(0, self.max_divide + 1):
            total_blocks += i ** 2

        return int(self.max_m_strokes / total_blocks)


    def _drawing_step_states(self):
        acc = self._compute_acc().item()
        print_fn = False
        if print_fn:
            print('iteration step %d, G_loss: %.5f, step_acc: %.5f, grid_scale: %d / %d, strokes: %d / %d'
                  % (self.step_id, self.G_loss.item(), acc,
                     self.m_grid, self.max_divide,
                     self.anchor_id + 1, self.m_strokes_per_block))
        vis2 = utils.patches2img(self.G_final_pred_canvas, self.m_grid).clip(min=0, max=1)
        if self.args.disable_preview:
            pass
        else:
            cv2.namedWindow('G_pred', cv2.WINDOW_NORMAL)
            cv2.namedWindow('input', cv2.WINDOW_NORMAL)
            cv2.imshow('G_pred', vis2[:,:,::-1])
            cv2.imshow('input', self.img_[:, :, ::-1])
            cv2.waitKey(1)


    def _render(self, v, path=None, canvas_start=None, save_jpgs=False, save_video=False):

        v = v[0,:,:self.rderr.d]   # if we add additional information, make sure to use only needed parms
        if self.args.keep_aspect_ratio:
            if self.input_aspect_ratio < 1:
                out_h = int(self.args.canvas_size * self.input_aspect_ratio)
                out_w = self.args.canvas_size
            else:
                out_h = self.args.canvas_size
                out_w = int(self.args.canvas_size / self.input_aspect_ratio)
        else:
            out_h = self.args.canvas_size
            out_w = self.args.canvas_size

        if save_video:
            video_writer = cv2.VideoWriter(
                path + '_animated.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20,
                (out_w, out_h))

        print('rendering canvas...')
        if canvas_start is None:
            self.rderr.create_empty_canvas()
        else:
            self.rderr.canvas = canvas_start

        alphas = []
        for i in range(v.shape[0]):  # for each stroke
            self.rderr.stroke_params = v[i, :]
            if self.rderr.check_stroke():
                alpha = self.rderr.draw_stroke()
                alphas.append(alpha)
            this_frame = self.rderr.canvas
            this_frame = cv2.resize(this_frame, (out_w, out_h), cv2.INTER_AREA)
            if save_jpgs:
                plt.imsave(os.path.join(path, str(i) + '.jpg'), this_frame)
            if save_video:
                video_writer.write((this_frame[:,:,::-1] * 255.).astype(np.uint8))

        final_rendered_image = np.copy(this_frame)
        # if save_jpgs:
        #     print('saving final rendered result...')
        #     plt.imsave(path + '_final.png', final_rendered_image)

        return final_rendered_image, np.concatenate(alphas)

    def get_checked_strokes(self, v):
        v = v[0, :, :]
        checked_strokes = []
        for i in range(v.shape[0]):
            if self.check_stroke(v[i,:]):
                checked_strokes.append(v[i, :][None, :])
        return np.concatenate(checked_strokes, axis=0)[None, :, :]   # restore the 1, n, parmas dimension for consistency

    @staticmethod
    def check_stroke(inp):
        """
        Copy and pasetd form renderder.py
        They have a threshold on the min size of the brushstorkes
        """

        r_ = max(inp[2], inp[3])   # check width and height, as in the original code
        if r_ > 0.025:
            return True
        else:
            return False

    def _save_stroke_params(self, v, path):

        d_shape = self.rderr.d_shape
        d_color = self.rderr.d_color
        d_alpha = self.rderr.d_alpha

        x_ctt = v[:, :, 0:d_shape]
        x_color = v[:, :, d_shape:d_shape+d_color]
        x_alpha = v[:, :, d_shape+d_color:d_shape+d_color+d_alpha]
        x_layer = v[:, :, d_shape+d_color+d_alpha:]

        path = os.path.join(path, 'strokes_params.npz')
        print(f'saving stroke parameters at {path}...')
        np.savez(path, x_ctt=x_ctt, x_color=x_color, x_alpha=x_alpha, x_layer=x_layer)


    def _shuffle_strokes_and_reshape(self, v):

        grid_idx = list(range(self.m_grid ** 2))
        random.shuffle(grid_idx)
        v = v[grid_idx, :, :]
        v = np.reshape(np.transpose(v, [1,0,2]), [-1, self.rderr.d])
        v = np.expand_dims(v, axis=0)

        return v, np.array(grid_idx)


    def clamp(self, val):
        # Modification, use a different clamp for width and height
        pos = torch.clamp(self.x_ctt.data[:, :, :2], 0.1, 1 - 0.1)
        size = torch.clamp(self.x_ctt.data[:, :, 2:4], 0.1, val)
        theta = torch.clamp(self.x_ctt.data[:, :, 4], 0.1, 1 - 0.1)

        # Put all back together
        self.x_ctt.data = torch.cat([pos, size, theta.unsqueeze(-1)], dim=-1)
        self.x_color.data = torch.clamp(self.x_color.data, 0, 1)
        self.x_alpha.data = torch.clamp(self.x_alpha.data, 0, 1)


    def train(self):
        # -------------------------------------------------------------------------------------------------------------
        # Set parameters
        # self.max_divide = args.max_divide
        # self.max_m_strokes = args.max_m_strokes

        # manually set the number of strokes, use more strokes at the beginning
        self.manual_strokes_per_block = self.args.manual_storkes_params
        self.m_strokes_per_block = None  # self.stroke_parser()

        self.max_divide = max(self.manual_strokes_per_block.keys())
        self.m_grid = 1

        self.img_path = self.args.img_path
        self.img_ = cv2.imread(self.args.img_path, cv2.IMREAD_COLOR)
        self.img_ = cv2.cvtColor(self.img_, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        self.input_aspect_ratio = self.img_.shape[0] / self.img_.shape[1]
        self.img_ = cv2.resize(self.img_, (self.net_G.out_size * self.max_divide,
                                           self.net_G.out_size * self.max_divide), cv2.INTER_AREA)

        # self._style_loss = loss.VGGStyleLoss(transfer_mode=1,
        #                                      resize=False)  # 0 to transfer only color, > 0 texture and color
        # self._content_loss = loss.VGGPerceptualLoss(resize=False)
        self.loss_dict = {'pixel_loss': [],
                          'style_loss': [],
                          'content_loss': []}
        #self.load_style_image()
        # --------------------------------------------------------------------------------------------------------------
        print('begin drawing...')

        clamp_schedule = self.args.clamp_schedule
        PARAMS = np.zeros([1, 0, self.rderr.d + 2], np.float32)  # +2 to save layer information

        if self.rderr.canvas_color == 'white':
            CANVAS_tmp = torch.ones([1, 3, 128, 128]).to(self.device)
        else:
            CANVAS_tmp = torch.zeros([1, 3, 128, 128]).to(self.device)

        for self.m_grid in self.manual_strokes_per_block.keys():

            self.img_batch = utils.img2patches(self.img_, self.m_grid, self.net_G.out_size).to(self.device)
            self.G_final_pred_canvas = CANVAS_tmp

            self.manual_set_number_strokes_per_block(self.m_grid)
            self.initialize_params()
            self.x_ctt.requires_grad = True
            self.x_color.requires_grad = True
            self.x_alpha.requires_grad = True
            utils.set_requires_grad(self.net_G, False)

            self.optimizer_x = optim.RMSprop([self.x_ctt, self.x_color, self.x_alpha], lr=self.lr, centered=True)

            self.step_id = 0
            for self.anchor_id in range(0, self.m_strokes_per_block):
                self.stroke_sampler(self.anchor_id)
                iters_per_stroke = int(500 / self.m_strokes_per_block)
                for i in range(iters_per_stroke):
                    self.G_pred_canvas = CANVAS_tmp

                    # update x
                    self.optimizer_x.zero_grad()
                    self.clamp(val=clamp_schedule[self.m_grid])

                    self._forward_pass()
                    self._drawing_step_states()
                    self._backward_x()

                    self.clamp(val=clamp_schedule[self.m_grid])

                    self.optimizer_x.step()
                    self.step_id += 1

            v = self._normalize_strokes(self.x)
            v, idx_grid = self._shuffle_strokes_and_reshape(v)

            # Add layer information
            layer_info = np.full((1, v.shape[1], 1), self.m_grid)
            grid_info = np.repeat(idx_grid, self.m_strokes_per_block)[None, :,
                        None]  # repeat for each storke, add dim 0 and -1
            v = np.concatenate([v, layer_info, grid_info], axis=-1)

            # Add on previous parmas
            PARAMS = np.concatenate([PARAMS, v], axis=1)
            CANVAS_tmp, _ = self._render(PARAMS, save_jpgs=False, save_video=False)
            CANVAS_tmp = utils.img2patches(CANVAS_tmp, self.m_grid + 1, self.net_G.out_size).to(self.device)

        PARAMS = self.get_checked_strokes(PARAMS)
        #final_rendered_image, alphas = self._render(PARAMS, save_jpgs=False, save_video=False)
        return PARAMS

    def inference(self, strokes, output_path=None, order=None, canvas_start=None, save_jpgs=False, save_video=False):

        if order is not None:
            strokes = strokes[:, order, :]

        if output_path is None:
            img, alphas = self._render(strokes, canvas_start=canvas_start)
            return img, alphas
        else:
            _ = self._render(strokes, path=output_path, canvas_start=canvas_start, save_jpgs=save_jpgs, save_video=save_video)