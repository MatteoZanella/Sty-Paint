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

        self.m_strokes_per_block = self.stroke_parser()

        self.m_grid = 1

        self.img_path = args.img_path
        self.img_ = cv2.imread(args.img_path, cv2.IMREAD_COLOR)
        self.img_ = cv2.cvtColor(self.img_, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        self.input_aspect_ratio = self.img_.shape[0] / self.img_.shape[1]
        self.img_ = cv2.resize(self.img_, (self.net_G.out_size * args.max_divide,
                                           self.net_G.out_size * args.max_divide), cv2.INTER_AREA)


    def stroke_parser(self):

        total_blocks = 0
        for i in range(0, self.max_divide + 1):
            total_blocks += i ** 2

        return int(self.max_m_strokes / total_blocks)


    def _drawing_step_states(self):
        acc = self._compute_acc().item()
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

    def _render(self, v, filename='original', save_jpgs=True, save_video=True):

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
            self.output_dir, filename)

        if save_video:
            video_writer = cv2.VideoWriter(
                file_name + '_animated.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 10,
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

        return final_rendered_image, np.concatenate(alphas)

    def get_checked_strokes(self, v):
        v = v[0, :, :]
        checked_strokes = []
        for i in range(v.shape[0]):
            self.rderr.stroke_params = v[i, :]
            if self.rderr.check_stroke():
                checked_strokes.append(v[i, :][None, :])
        return np.concatenate(checked_strokes, axis=0)[None, :, :]

