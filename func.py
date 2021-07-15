from renderer import Renderer
from renderer import _normalize
import numpy as np
import matplotlib.pyplot as plt
import cv2
import utils


class RenderWithTransparnecy(Renderer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.transparency = []
        self.i = 0

    def _reset_transparency(self):
        self.transparency = []

    def _draw_oilpaintbrush(self):

        # xc, yc, w, h, theta, R0, G0, B0, R2, G2, B2, A
        x0, y0, w, h, theta = self.stroke_params[0:5]
        R0, G0, B0, R2, G2, B2, ALPHA = self.stroke_params[5:]
        x0 = _normalize(x0, self.CANVAS_WIDTH)
        y0 = _normalize(y0, self.CANVAS_WIDTH)
        w = (int)(1 + w * self.CANVAS_WIDTH)
        h = (int)(1 + h * self.CANVAS_WIDTH)
        theta = np.pi * theta

        if w * h / (self.CANVAS_WIDTH ** 2) > 0.1:
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

        self.foreground = np.array(self.foreground, dtype=np.float32) / 255.
        self.stroke_alpha_map = np.array(self.stroke_alpha_map, dtype=np.float32) / 255.
        self.transparency.append(np.expand_dims(self.stroke_alpha_map[:, :, 0], 0).astype('bool'))

        self.canvas = self._update_canvas()


def check_storkes(s, renderer):
    s = s[0, :, :]

    good = []
    for i in range(s.shape[0]):
        renderer.stroke_params = s[i, :]
        if renderer.check_stroke():
            good.append(s[i, :][None, :])
    return np.concatenate(good, axis=0)[None, :, :]


def main(rderr, v, file_name, save_jpgs=False, save_video=False, save_all_frames=False):
    v = v[0, :, :]

    out_h = 512
    out_w = 512

    if save_video:
        video_writer = cv2.VideoWriter(
            file_name + '_animated.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 10,
            (out_w, out_h))

    print('rendering canvas...')

    rderr.create_empty_canvas()

    for i in range(v.shape[0]):  # for each stroke
        rderr.stroke_params = v[i, :]
        if rderr.check_stroke():
            rderr.draw_stroke()
        this_frame = rderr.canvas
        this_frame = cv2.resize(this_frame, (out_w, out_h), cv2.INTER_AREA)
        if save_all_frames:
            plt.imsave(file_name + '_rendered_stroke_' + str((i + 1)).zfill(4) +
                       '.png', this_frame)
        if save_video:
            video_writer.write((this_frame[:, :, ::-1] * 255.).astype(np.uint8))

    final_rendered_image = np.copy(this_frame)
    if save_jpgs:
        print('saving final rendered result...')
        plt.imsave(file_name + '_final.png', final_rendered_image)

    return final_rendered_image