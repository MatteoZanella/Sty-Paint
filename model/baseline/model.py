from . import network, render, render_utils
import paddle
import paddle.nn as nn
import numpy as np

'''
Code adapted from https://github.com/wzmsltw/PaintTransformer
'''
def _normalize(x, width):
    return (int)(x * (width - 1) + 0.5)

class PaintTransformer:

    def __init__(self, model_path, config):
        self.stroke_num = 8
        self.input_size = config.canvas_size
        self.model_path = model_path

        # Create and load net
        paddle.set_device('gpu')
        self.net_g = network.Painter(5, self.stroke_num, 256, 8, 3, 3)
        self.net_g.set_state_dict(paddle.load(model_path))
        self.net_g.eval()
        for param in self.net_g.parameters():
            param.stop_gradient = True

        #  Load brushes
        brush_large_vertical = render_utils.read_img(config.brush_paths["large_vertical"], 'L')
        brush_large_horizontal = render_utils.read_img(config.brush_paths["large_horizontal"], 'L')
        self.meta_brushes = paddle.concat([brush_large_vertical, brush_large_horizontal], axis=0)

    def get_ctx(self, ctx):
        # Location of the last stroke
        x_start, y_start = ctx[0, -1, :2]
        x_start = _normalize(x_start, self.input_size)
        y_start = _normalize(y_start, self.input_size)

        # Select window size based on average stroke area
        area = ctx[:, :, 2] * ctx[:, :, 3]   # h*w
        area = area.mean()
        if area < 0.004:
            windows_size = 32
        elif area < 0.01:
            windows_size = 64
        else:
            windows_size = 128
        # print(f'Area: {area}, ws: {windows_size}')

        return (x_start, y_start), windows_size

    def generate(self, data):
        original = data['img']
        canvas_start = data['canvas']
        strokes_ctx = data['strokes_ctx']

        bs = original.shape[0]
        out = np.empty([bs, 8, 11])
        for b in range(bs):
            res = self.main(original[b][None], canvas_start[b][None], strokes_ctx[b][None])
            out[b] = res
        return out.astype('float32')

    def main(self, original, canvas_start, strokes_ctx):
        assert original.size(2) == canvas_start.size(2) == self.input_size

        # Crop input
        st_point, ws = self.get_ctx(strokes_ctx)
        x1, x2, y1, y2 = render_utils.get_bbox(st_point, ws, self.input_size)

        # crop
        original = original[:, :, y1 :y2, x1 :x2]
        canvas_start = canvas_start[:, :, y1 :y2, x1 :x2]

        original_img = render_utils.torch_to_paddle(original)
        canvas_start = render_utils.torch_to_paddle(canvas_start)

        # Predict strokes
        sparms, dec, idx = self.predict(original_img, canvas_start)

        # Refactor sparams to match stylized neural painter renderer
        n = sparms.shape[0]
        sparms = np.concatenate((sparms, sparms[:, -3:]), axis=-1)   # replicate the color, add a 0 for transparency, note that it won't be used
        sparms[:, 0] = (sparms[:, 0] * ws + x1) / self.input_size
        sparms[:, 1] = (sparms[:, 1] * ws + y1) / self.input_size
        sparms[:, 2] = (sparms[:, 2] * ws) / self.input_size
        sparms[:, 3] = (sparms[:, 3] * ws) / self.input_size

        # Reorder storkes based on idx, unsqueeze dim=0
        sparms = sparms[idx, :][None]


        return sparms

    def predict(self, original_img, canvas_status) :
        patch_size = 32
        stroke_num = 8
        H, W = original_img.shape[-2 :]

        dilation = render_utils.Dilation2d(m=1)
        erosion = render_utils.Erosion2d(m=1)

        # Do a single prediction with window type A
        layer = 0
        layer_size = patch_size

        with paddle.no_grad() :
            # * ----- read in image and init canvas ----- *#
            final_result = canvas_status

            img = nn.functional.interpolate(original_img, (layer_size, layer_size))
            result = nn.functional.interpolate(final_result, (layer_size, layer_size))
            img_patch = nn.functional.unfold(img, [patch_size, patch_size], strides=[patch_size, patch_size])
            result_patch = nn.functional.unfold(result, [patch_size, patch_size],
                                                strides=[patch_size, patch_size])
            h = (img.shape[2] - patch_size) // patch_size + 1
            w = (img.shape[3] - patch_size) // patch_size + 1
            render_size_y = int(1.25 * H // h)
            render_size_x = int(1.25 * W // w)

            # * -------------------------------------------------------------*#
            # * -------------generate strokes on window type A---------------*#
            # * -------------------------------------------------------------*#

            # Note: we use only type A
            param, decision = self.stroke_net_predict(img_patch, result_patch, patch_size, stroke_num)

            # expand_img = original_img
            sparam, idx, decisions = \
                self.get_single_layer_lists(param, decision, original_img, render_size_x, render_size_y, h, w,
                                       self.meta_brushes, dilation, erosion, stroke_num)
        return sparam, decisions, idx

    def get_single_layer_lists(self, param, decision, ori_img, render_size_x, render_size_y, h, w, meta_brushes, dilation,
                               erosion, stroke_num) :
        """
        get_single_layer_lists
        """
        valid_foregrounds = render_utils.param2stroke(param[:, :], render_size_y, render_size_x, meta_brushes)

        valid_alphas = (valid_foregrounds > 0).astype('float32')
        valid_foregrounds = valid_foregrounds.reshape([-1, stroke_num, 1, render_size_y, render_size_x])
        valid_alphas = valid_alphas.reshape([-1, stroke_num, 1, render_size_y, render_size_x])

        temp = [dilation(valid_foregrounds[:, i, :, :, :]) for i in range(stroke_num)]
        valid_foregrounds = paddle.stack(temp, axis=1)
        valid_foregrounds = valid_foregrounds.reshape([-1, 1, render_size_y, render_size_x])

        temp = [erosion(valid_alphas[:, i, :, :, :]) for i in range(stroke_num)]
        valid_alphas = paddle.stack(temp, axis=1)
        valid_alphas = valid_alphas.reshape([-1, 1, render_size_y, render_size_x])

        patch_y = 4 * render_size_y // 5
        patch_x = 4 * render_size_x // 5

        img_patch = ori_img.reshape([1, 3, h, ori_img.shape[2] // h, w, ori_img.shape[3] // w])
        img_patch = img_patch.transpose([0, 2, 4, 1, 3, 5])[0]

        xid_list = []
        yid_list = []
        error_list = []

        for flag_idx, flag in enumerate(decision.cpu().numpy()) :
            if flag :
                flag_idx = flag_idx // stroke_num
                x_id = flag_idx % w
                flag_idx = flag_idx // w
                y_id = flag_idx % h
                xid_list.append(x_id)
                yid_list.append(y_id)

        inner_fores = valid_foregrounds[:, :, render_size_y // 10 :9 * render_size_y // 10,
                      render_size_x // 10 :9 * render_size_x // 10]
        inner_alpha = valid_alphas[:, :, render_size_y // 10 :9 * render_size_y // 10,
                      render_size_x // 10 :9 * render_size_x // 10]
        inner_fores = inner_fores.reshape([h * w, stroke_num, 1, patch_y, patch_x])
        inner_alpha = inner_alpha.reshape([h * w, stroke_num, 1, patch_y, patch_x])
        inner_real = img_patch.reshape([h * w, 3, patch_y, patch_x]).unsqueeze(1)

        R = param[:, 5]
        G = param[:, 6]
        B = param[:, 7]  # , G, B = param[5:]
        R = R.reshape([-1, stroke_num]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        G = G.reshape([-1, stroke_num]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        B = B.reshape([-1, stroke_num]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        error_R = R * inner_fores - inner_real[:, :, 0 :1, :, :]
        error_G = G * inner_fores - inner_real[:, :, 1 :2, :, :]
        error_B = B * inner_fores - inner_real[:, :, 2 :3, :, :]
        error = paddle.abs(error_R) + paddle.abs(error_G) + paddle.abs(error_B)

        error = error * inner_alpha
        error = paddle.sum(error, axis=(2, 3, 4)) / paddle.sum(inner_alpha, axis=(2, 3, 4))
        #error_list = error.reshape([-1]).numpy()[decision.numpy()]   # keep all the predictions
        error_list =  error.reshape([-1]).numpy()
        error_list = list(error_list)

        sort_list = list(np.argsort(error_list))

        return param.numpy(), sort_list, decision.numpy()

    def stroke_net_predict(self, img_patch, result_patch, patch_size, stroke_num) :
        """
        stroke_net_predict
        """
        img_patch = img_patch.transpose([0, 2, 1]).reshape([-1, 3, patch_size, patch_size])
        result_patch = result_patch.transpose([0, 2, 1]).reshape([-1, 3, patch_size, patch_size])
        # *----- Stroke Predictor -----*#
        shape_param, stroke_decision = self.net_g(img_patch, result_patch)
        stroke_decision = (stroke_decision > 0).astype('float32')
        # *----- sampling color -----*#
        grid = shape_param[:, :, :2].reshape([img_patch.shape[0] * stroke_num, 1, 1, 2])
        img_temp = img_patch.unsqueeze(1).tile([1, stroke_num, 1, 1, 1]).reshape([
            img_patch.shape[0] * stroke_num, 3, patch_size, patch_size])
        color = nn.functional.grid_sample(img_temp, 2 * grid - 1, align_corners=False).reshape([
            img_patch.shape[0], stroke_num, 3])
        stroke_param = paddle.concat([shape_param, color], axis=-1)

        param = stroke_param.reshape([-1, 8])
        decision = stroke_decision.reshape([-1]).astype('bool')
        param[:, :2] = param[:, :2] / 1.25 + 0.1
        param[:, 2 :4] = param[:, 2 :4] / 1.25
        return param, decision

