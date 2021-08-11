from painter import *
from renderer import _normalize, Renderer

import torch

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NewRenderer(Renderer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _draw_oilpaintbrush(self):

        # xc, yc, w, h, theta, R0, G0, B0, R2, G2, B2, A
        x0, y0, w, h, theta = self.stroke_params[0:5]
        R0, G0, B0, R2, G2, B2, ALPHA = self.stroke_params[5:]
        x0 = _normalize(x0, self.CANVAS_WIDTH)
        y0 = _normalize(y0, self.CANVAS_WIDTH)
        w = (int)(1 + w * self.CANVAS_WIDTH)
        h = (int)(1 + h * self.CANVAS_WIDTH)
        theta = np.pi*theta

        if w * h / (self.CANVAS_WIDTH**2) > 0.1:
            if h > w:
                brush = self.brush_large_vertical
            else:
                brush = self.brush_large_horizontal
        else:
            if h > w:
                brush = self.brush_small_vertical
            else:
                brush = self.brush_small_horizontal
        self.foreground, self.stroke_alpha_map = utils.create_transformed_brush(
            brush, self.CANVAS_WIDTH, self.CANVAS_WIDTH,
            x0, y0, w, h, theta, R0, G0, B0, R2, G2, B2)

        if not self.train:
            self.foreground = cv2.dilate(self.foreground, np.ones([2, 2]))
            self.stroke_alpha_map = cv2.erode(self.stroke_alpha_map, np.ones([2, 2]))

        self.foreground = np.array(self.foreground, dtype=np.float32)/255.
        self.stroke_alpha_map = np.array(self.stroke_alpha_map, dtype=np.float32)/255.
        self.canvas = self._update_canvas()

        return np.expand_dims(self.stroke_alpha_map[:, :, 0], 0).astype('bool')


class NewPainter(PainterBase):

    def __init__(self, args):
        super(NewPainter, self).__init__(args=args)

        self.rderr = NewRenderer(renderer=args.renderer,
                                       CANVAS_WIDTH=args.canvas_size, canvas_color=args.canvas_color)

        self.max_divide = args.max_divide
        self.max_m_strokes = args.max_m_strokes

        # allocate more storkes at the beginning
        self.manual_strokes_per_block = {2:30, 3:20, 4:15, 5:10} ##{1:9, 2:9, 3:9, 4:9, 5:9} #
        self.m_strokes_per_block = None #self.stroke_parser()

        self.m_grid = 1

        self.img_path = args.img_path
        self.img_ = cv2.imread(args.img_path, cv2.IMREAD_COLOR)
        self.img_ = cv2.cvtColor(self.img_, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        self.input_aspect_ratio = self.img_.shape[0] / self.img_.shape[1]
        self.img_ = cv2.resize(self.img_, (self.net_G.out_size * args.max_divide,
                                           self.net_G.out_size * args.max_divide), cv2.INTER_AREA)

        self._style_loss = loss.VGGStyleLoss(transfer_mode=1,
                                             resize=False)  # 0 to transfer only color, > 0 texture and color
        self._content_loss = loss.VGGPerceptualLoss(resize=False)
        self.loss_dict = {'pixel_loss': [],
                          'style_loss': [],
                          'content_loss': []}

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

        with torch.no_grad():
            canvas = utils.patches2img(
                self.G_final_pred_canvas, self.m_grid, to_numpy=False).to(device)
            style_img = utils.patches2img(self.img_batch, self.m_grid, to_numpy=False).to(device)

            assert canvas.shape == style_img.shape
            content_loss = self._content_loss(canvas, style_img)
            style_loss = self._style_loss(canvas, style_img)

            """
            style_loss = 0.0
            content_loss = 0.0
            N = self.G_final_pred_canvas.shape[0]
            for i in range(N):
                gi = self.G_final_pred_canvas[i].unsqueeze(0)
                ii = self.img_batch[i].unsqueeze(0)
                style_loss += self._style_loss(gi, ii)
                content_loss += self._content_loss(gi, ii)
            style_loss /= N
            content_loss /= N
            #style_loss = self._style_loss(canvas, self.style_img)
            #content_loss = self._content_loss(canvas, self.style_img)
            """
        self.loss_dict['pixel_loss'].append(self.G_loss.item())
        self.loss_dict['style_loss'].append(style_loss.item())
        self.loss_dict['content_loss'].append(content_loss.item())

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
        print_fn = True
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


    def _render(self, v, path, save_jpgs=True, save_video=True):

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
                path + '_animated.avi', cv2.VideoWriter_fourcc(*'MPEG'), 20,
                (out_w, out_h))

        print('rendering canvas...')
        self.rderr.create_empty_canvas()
        alphas = []
        for i in range(v.shape[0]):  # for each stroke
            self.rderr.stroke_params = v[i, :]
            if self.rderr.check_stroke():
                alpha = self.rderr.draw_stroke()
                alphas.append(alpha)
            this_frame = self.rderr.canvas
            this_frame = cv2.resize(this_frame, (out_w, out_h), cv2.INTER_AREA)
            if save_jpgs:
                plt.imsave(path + '_rendered_stroke_' + str((i+1)).zfill(4) +
                           '.png', this_frame)
            if save_video:
                video_writer.write((this_frame[:,:,::-1] * 255.).astype(np.uint8))

        if save_jpgs:
            print('saving input photo...')
            out_img = cv2.resize(self.img_, (out_w, out_h), cv2.INTER_AREA)
            plt.imsave(path + '_input.png', out_img)

        final_rendered_image = np.copy(this_frame)
        if save_jpgs:
            print('saving final rendered result...')
            plt.imsave(path + '_final.png', final_rendered_image)

        return final_rendered_image, np.concatenate(alphas)

    def get_checked_strokes(self, v):
        v = v[0, :, :]
        checked_strokes = []
        for i in range(v.shape[0]):
            if self.check_stroke(v[i,:]):
                checked_strokes.append(v[i, :][None, :])
        return np.concatenate(checked_strokes, axis=0)[None, :, :]   # restore the 1, n, parmas dimension for consistency

    def check_stroke(self, inp):
        """
        Copy and pasetd form renderder.py
        They have a threshold on the min size of the brushstorkes
        """

        r_ = max(inp[2], inp[3])   # check wifth and height, as in the original code
        if r_ > 0.025:
            return True
        else:
            return False

    def _save_stroke_params(self, v):

        d_shape = self.rderr.d_shape
        d_color = self.rderr.d_color
        d_alpha = self.rderr.d_alpha

        x_ctt = v[:, :, 0:d_shape]
        x_color = v[:, :, d_shape:d_shape+d_color]
        x_alpha = v[:, :, d_shape+d_color:d_shape+d_color+d_alpha]
        x_layer = v[:, :, d_shape+d_color+d_alpha:]
        print('saving stroke parameters...')
        np.savez(os.path.join(self.output_dir, 'strokes_params.npz'), x_ctt=x_ctt,
                 x_color=x_color, x_alpha=x_alpha, x_layer=x_layer)


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

