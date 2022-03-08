import cv2
import argparse
import numpy as np

from dataset_acquisition.decomposition.painter import Painter
from dataset_acquisition.decomposition.utils import load_painter_config
from model import build_model
import torch


def void(val):
    pass

class InteractiveNP:

    def __init__(self, args):

        self.img_size = args.img_size
        img = cv2.cvtColor(cv2.imread(args.img_path), cv2.COLOR_BGR2RGB)
        self.img = cv2.resize(img, (args.img_size, args.img_size))
        self.canvas = np.zeros_like(self.img)
        self.worksapace = np.zeros_like(self.img)
        self.renderer = Painter(args=load_painter_config(args.renderer_config))


        self.img_window_name = 'Image'
        self.canvas_window_name = 'Canvas'
        self.workspace_window_name = 'Workspace'

        # Trackbar config
        self.max_w = 100
        self.max_h = 100
        self.max_theta = 100
        self.max_K = 8

        #
        self.S = torch.empty(1, 1, 8)
        self.curr_author = None
        self.author = []
        self.frames = []


        #
        self.set_up()

        #
        self.ctx = 10
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        self.model = build_model(ckpt["config"])
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()


    def set_up(self):

        # Reference Image
        cv2.namedWindow(self.img_window_name)
        cv2.imshow(self.img_window_name, cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR))

        # Canvas
        cv2.namedWindow(self.canvas_window_name)
        cv2.imshow(self.canvas_window_name, self.canvas)

        # Workspace
        cv2.namedWindow(self.workspace_window_name)
        cv2.imshow(self.workspace_window_name, self.worksapace)
        cv2.setMouseCallback(self.workspace_window_name, self.pointer)
        cv2.createTrackbar('width', self.workspace_window_name, 0, self.max_w, void)
        cv2.createTrackbar('height', self.workspace_window_name, 0, self.max_h, void)
        cv2.createTrackbar('theta', self.workspace_window_name, 0, self.max_theta, void)
        cv2.createTrackbar('K', self.workspace_window_name, 0, self.max_K, void)

    def pointer(self, event, x, y, flags, params) :
        if event == cv2.EVENT_LBUTTONDOWN :
            self.points = [x, y]

    def on_trackbar(self):
        w = cv2.getTrackbarPos('width', self.workspace_window_name)
        h = cv2.getTrackbarPos('height', self.workspace_window_name)
        theta = cv2.getTrackbarPos('theta', self.workspace_window_name)
        k = cv2.getTrackbarPos('K', self.workspace_window_name)

        w, h, theta = w / self.max_w, h / self.max_h, theta / self.max_theta

        return dict(w=w, h=h, theta=theta, k=k)

    def get_params(self) :
        x, y = self.points
        R0, G0, B0 = self.img[int(y), int(x)] / 255.

        slider_params = self.on_trackbar()
        x = x / args.img_size
        y = y / args.img_size

        return np.array([x, y, slider_params["w"], slider_params["h"], slider_params["theta"],
                         R0.item(), G0.item(), B0.item()])[None, None, :]

    def update_canvas(self) :

        N = self.params.shape[1]

        for j in range(N):
            s = self.params[:, j, :][:, None, :]
            foreground, alpha_map = self.renderer.inference(s)

            foreground = cv2.dilate(np.uint8(foreground * 255.), np.ones([2, 2]))
            alpha_map = np.repeat((alpha_map * 255.0)[0, :, :, None], 3, axis=-1)
            alpha_map = cv2.erode(alpha_map, np.ones([2, 2]))

            # Morphology
            foreground = np.array(foreground, dtype=np.float32) / 255.
            alpha_map = np.array(alpha_map, dtype=np.float32) / 255.

            self.canvas = foreground * alpha_map + self.canvas * (1 - alpha_map)

    def preprocess(self):

        img = torch.tensor(cv2.resize(self.img, (256, 256)) / 255).permute(2, 0, 1).unsqueeze(0).float()
        canvas = torch.tensor(cv2.resize(self.canvas, (256, 256)) / 255).permute(2, 0, 1).unsqueeze(0).float()

        s_c = self.S[:, -self.ctx:, :].float()
        s_t = torch.rand_like(s_c)


        print(img.shape)
        print(canvas.shape)
        print(s_c.shape)

        return dict(img=img, canvas=canvas, strokes_ctx=s_c, strokes_seq=s_t)


    def user(self):
        print('User Painting...')
        if len(self.points) == 0 :
            return
        self.params = self.get_params()
        foreground, alpha_map = self.renderer.inference(self.params)
        cv2.imshow(self.workspace_window_name, cv2.cvtColor(np.uint8(foreground * 255.0), cv2.COLOR_RGB2BGR))
        self.curr_author = 'user'


    def tool(self):
        print('Model suggesting...')
        batch = self.preprocess()
        with torch.no_grad():
            p = self.model.generate(batch)
            self.params = p["fake_data_random"]
        foreground, alpha_map = self.renderer.inference(self.params)
        cv2.imshow(self.workspace_window_name, cv2.cvtColor(np.uint8(foreground * 255.0), cv2.COLOR_RGB2BGR))
        self.curr_author = 'model'
        '''
        K = cv2.getTrackbarPos('K', self.workspace_window_name)
        if K == 0 :
            print('==> No good suggestion, keep on painting')
            return
        else:
            print(f'==> The user selected: {K} strokes')
            import pdb
            pdb.set_trace()
            self.params = self.params[:, :K, :]
            self.foreground, self.alpha_map = self.renderer.inference(self.params)
        '''

    def draw(self):
        print('Drawing on the canvas...')
        if self.params is None:
            return
        self.update_canvas()
        cv2.imshow(self.canvas_window_name, cv2.cvtColor(np.uint8(self.canvas * 255.0), cv2.COLOR_RGB2BGR))
        if torch.is_tensor(self.params):
            self.S = torch.cat((self.S, self.params.cpu()), dim=1)
        else:
            self.S = torch.cat((self.S, torch.tensor(self.params)), dim=1)

        print(f'** Author: {self.curr_author} **')

    def reset(self):
        self.params = None

    def interact(self):

        self.params = None
        while True :
            # display the image and wait for a keypress
            key = cv2.waitKey(1) & 0xFF
            # if the 'd' key is pressed, draw the stroke
            if key == ord("p") :
                self.user()
            elif key == ord("m"):
                self.tool()
            elif key == ord("r") :
                self.reset()
            elif key == ord("d") :
                self.draw()
            elif key == ord("e"):
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", default='/Users/eliap/Desktop/INP - ECCV 2022/test_images/chicago.jpg')
    parser.add_argument("--img_size", type=int, default=512)


    parser.add_argument('--checkpoint',
                        default='/Users/eliap/Downloads/final_models_checkoints/ade_final_model-vae-pos2.0-gt_col0.25-ref_col0.25-kl0.00025/latest.pth.tar')
    parser.add_argument('--renderer_config', default='/Users/eliap/Projects/brushstrokes-generation/configs/decomposition/painter_config_local.yaml')
    args = parser.parse_args()

    INP = InteractiveNP(args)
    INP.interact()
    cv2.destroyAllWindows()


