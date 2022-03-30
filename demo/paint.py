from tkinter import *
from PIL import ImageTk, Image
from tkinter import colorchooser
import tkinter.font as TkFont
import cv2
import numpy as np
import copy
import math
import argparse

import torch
from dataset_acquisition.decomposition.painter import Painter
from dataset_acquisition.decomposition.utils import load_painter_config
from evaluation.tools import check_strokes
from model import build_model

# Main Parent window class
def load_button_iamges(path, size):
    image = Image.open(path)
    image = image.resize((size, size), Image.ANTIALIAS)
    return ImageTk.PhotoImage(image)


def normalize(x):
    return np.uint8(x * 255.)


class MovableWindow:
    """
        CANVAS WHICH IS MOVABLE WHEN CLICKED AND DRAGGED
    """

    def __init__(self, parent):
        self.parent = parent
        # create a canvas which is movable
        self.Holder = Frame(self.parent, width=260, height=20, bg=BG_COLOR)
        self.Holder.grid(row=0, column=1, ipady=2)
        self.MovableCanvas = Canvas(self.Holder, width=260, height=25,
                                    bg=BG_COLOR, highlightthickness=0)
        self.MovableCanvas.bind('<Button-1>', self.ClickTopLevel)
        self.MovableCanvas.bind('<B1-Motion>', self.DragTopLevel)
        self.MovableCanvas.grid(row=0, column=1, sticky=W)

        # CLOSE WINDOW BUTTON
        self.closeXmarker = ImageTk.PhotoImage(file="icons/cc.png")
        self.closeBrushWin = Button(self.Holder, width=20, height=20, bd=0, bg=BG_COLOR, command=self.parent.withdraw,
                                    cursor='hand2', activebackground=BG_COLOR, highlightthickness=0)
        self.closeBrushWin.config(image=self.closeXmarker)
        self.closeBrushWin.grid(row=0, column=2, sticky=E, padx=7)

    def ClickTopLevel(self, event):
        self.TopLevelXPos, self.TopLevelYPos = event.x, event.y

    def DragTopLevel(self, event):
        # print(event)
        self.childWin = self.parent
        x = self.childWin.winfo_pointerx() - self.TopLevelXPos
        y = self.childWin.winfo_pointery() - self.TopLevelYPos
        # self.childWin.config(cursor = 'fleur')
        self.childWin.geometry('+{x}+{y}'.format(x=x, y=y))


class Pointer:
    def __init__(self):
        self.start = None
        self.end = None

    def check(self):
        return self.start is not None and self.end is not None

    def reset(self):
        self.start = None
        self.end = None

    def compute_h(self):
        if not self.check():
            return
        else:
            h = math.dist(self.start, self.end)
        return h

    def compute_center(self):
        if not self.check():
            return
        else:
            xc = (self.start[0] + self.end[0]) / 2
            yc = (self.start[1] + self.end[1]) / 2
        return (xc, yc)

    def compute_angle(self):

        if not self.check():
            return
        else:
            delta_x = np.array(self.end[0] - self.start[0])
            delta_y = np.array(- (self.end[1] - self.start[1]))

            theta = (np.arctan2(delta_x, delta_y) % np.pi) / np.pi

            return theta


class WindowApp(MovableWindow):

    def __init__(self, master, args):
        self.master = master
        self.master.geometry(f"{SIZE[0]}x{SIZE[1]}")
        self.master.config(bg=BG_COLOR)
        self.master.title('I-Paint')
        self.master.resizable(0, 0)

        self.img_size = 512
        self.bttn_size = 45
        self.font_size = TkFont.Font(family="Helvetica", size=20, weight="bold")
        self.fg_color = "black"

        self.c1 = self.img_size / 2
        self.c2 = self.img_size / 2
        self.params = None

        # Canvas
        self.DrawCanvas = Canvas(self.master, width=SIZE[0], height=self.img_size, bg='#333')
        self.DrawCanvas.grid(row=0)
        self.path = args.img_path
        self.img = cv2.cvtColor(cv2.imread(self.path), cv2.COLOR_BGR2RGB)
        self.img = cv2.resize(self.img, (self.img_size, self.img_size))
        self.workspace = cv2.addWeighted(self.img, 0.2, np.zeros_like(self.img), 0.8, 0) / 255.
        self.canvas = np.zeros_like(self.workspace)

        # image converted to photoimg object
        self.img_tk = ImageTk.PhotoImage(Image.fromarray(normalize(self.workspace)))
        self.DrawCanvasContainer = self.DrawCanvas.create_image(self.c1, self.c2, image=self.img_tk)

        # Model and Renderer
        self.renderer = Painter(args=load_painter_config(args.painter_config))
        self.S = torch.empty(1, 1, 8)
        self.ctx = 10
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        self.model = build_model(ckpt["config"])
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        # ==================================================================================
        # Commands section
        self.CommandFrame = Frame(self.master, width=SIZE[0], height=120, bg=BG_COLOR)
        self.CommandFrame.grid(row=1)

        # ==========
        # Draw, Predict, Cancel
        self.ToolBoxFrame = Frame(self.CommandFrame, width=SIZE[0], height=100, bg=BG_COLOR)
        self.ToolBoxFrame.grid(row=0, column=0, padx=0, pady=5, sticky=NW)

        # Draw
        self.drawLabel = Label(self.ToolBoxFrame, text="Draw", bg=BG_COLOR, fg=self.fg_color, font=self.font_size,
                               justify=LEFT, anchor="w")
        self.drawLabel.grid(row=0, column=0, padx=35, sticky=W)
        self.drawBtn = Button(self.ToolBoxFrame, text='',
                              command=self.draw_on_canvas,
                              width=self.bttn_size,
                              height=self.bttn_size,
                              bd=0,
                              bg=BG_COLOR,
                              cursor="hand2")
        self.drawBtnImg = load_button_iamges('icons/paint-brush.png', size=self.bttn_size)
        self.drawBtn.config(image=self.drawBtnImg)
        self.drawBtn.grid(row=1, column=0, padx=34)

        # Predict
        self.predictLabel = Label(self.ToolBoxFrame, text="I-Paint", bg=BG_COLOR, fg=self.fg_color,
                                  font=TkFont.Font(family="Helvetica", size=20, weight="bold", slant="italic"),
                                  justify=LEFT, anchor="w")
        self.predictLabel.grid(row=0, column=1, padx=30, pady=0, sticky=W)
        self.predictBttn = Button(self.ToolBoxFrame, text='',
                                  width=self.bttn_size,
                                  height=self.bttn_size,
                                  bd=0,
                                  bg=BG_COLOR,
                                  cursor="hand2",
                                  command=self.model_prediction)
        self.predictBttnImg = load_button_iamges('icons/neural_black.png', size=self.bttn_size)
        self.predictBttn.config(image=self.predictBttnImg)
        self.predictBttn.grid(row=1, column=1, padx=30)

        # Cancel
        self.cancelLabel = Label(self.ToolBoxFrame, text="Cancel", bg=BG_COLOR, fg=self.fg_color, font=self.font_size,
                                 justify=LEFT, anchor="w")
        self.cancelLabel.grid(row=0, column=2, padx=30, pady=5, sticky=W)
        self.cancelBtn = Button(self.ToolBoxFrame, text="",
                                width=self.bttn_size,
                                height=self.bttn_size,
                                bd=0,
                                bg=BG_COLOR,
                                cursor='hand2',
                                command=self.cancel)
        self.cancelBtnImg = load_button_iamges('icons/close.png', size=int(0.7 * self.bttn_size))
        self.cancelBtn.config(image=self.cancelBtnImg)
        self.cancelBtn.grid(row=1, column=2, padx=30)

        # Color picker and palette
        self.editWin = Frame(self.CommandFrame, width=100, height=50, bg=BG_COLOR)
        self.editWin.grid(row=1, column=0, padx=0, pady=10, sticky=W)

        # THIKNESS SLIDER
        self.thicknessLabel = Label(self.editWin, text="Thickness", bg=BG_COLOR, fg=self.fg_color,
                                    font=TkFont.Font(family="Helvetica", size=18, weight="bold"), justify=RIGHT,
                                    anchor="w")
        self.thicknessLabel.grid(row=0, column=1, pady=10, padx=10, sticky=W)

        self.BrushThickness = Scale(self.editWin, from_=0, to=100, orient=HORIZONTAL, width=7, bd=0, bg='#24272b',
                                    cursor='hand2',
                                    fg="white",
                                    length=70)
        self.BrushThickness.grid(row=0, column=2, padx=0, sticky=W)

        # COLOR PALETTE
        self.ColorPaletteLabel = Label(self.editWin, text='Color Picker', fg=self.fg_color, bg=BG_COLOR,
                                       font=TkFont.Font(family="Helvetica", size=18, weight="bold"), justify=RIGHT,
                                       anchor="w")
        self.ColorPaletteLabel.grid(row=0, column=3, pady=5, padx=5, sticky=W)
        self.colorBttn = Button(self.editWin, text="",
                                width=self.bttn_size,
                                height=self.bttn_size,
                                bd=0,
                                bg=BG_COLOR,
                                cursor='hand2',
                                command=self.colorpicker)
        self.colorBttnImg = load_button_iamges('icons/color-circle.png', size=self.bttn_size)
        self.colorBttn.config(image=self.colorBttnImg)
        self.colorBttn.grid(row=0, column=4)

        # ==============================================================================================================
        # Action to section
        # Pointer
        self.Pointer = Pointer()
        self.DrawCanvas.bind('<ButtonPress-1>', self.draw_line)
        self.DrawCanvas.bind('<ButtonRelease-1>', self.draw_line)
        self.DrawCanvas.bind('<Motion>', self.draw_line)
        self.BrushThickness.bind('<Motion>', self.change_thickness)

    # Define actions
    def colorpicker(self):
        color_code = colorchooser.askcolor(title="Choose color")
        rgb = color_code[0]
        if rgb is not None:
            self.set_params(rgb=np.uint8(rgb))
            self.show()

    def set_params(self, rgb=None):
        if not self.Pointer.check():
            self.params = None
            return

        center = self.Pointer.compute_center()
        h = self.Pointer.compute_h() / self.img_size
        theta = self.Pointer.compute_angle()

        # Thickness
        thick = self.BrushThickness.get()
        w = (thick / 100) * h

        # Color, if provided by color picker otherwise select from image
        if rgb is None:
            r, g, b = self.img[int(center[1]), int(center[0])] / 255.0
        else:
            r, g, b = rgb / 255.0

        # set params
        self.params = np.array([center[0] / self.img_size, center[1] / self.img_size, w, h, theta, r, g, b])[None, None,
                      :]

    def show(self, highlight_border=False, color=(1, 0, 0)):
        if self.params is None:
            return
        output = self.renderer._render(check_strokes(self.params),
                                       canvas_start=self.workspace,
                                       highlight_border=highlight_border,
                                       color_border=color)[0]
        self.tk_show(normalize(output))

    def tk_show(self, img):
        self.DrawCanvas.delete(ALL)
        self.img_tk = ImageTk.PhotoImage(Image.fromarray(img))
        self.DrawCanvas.create_image(self.c1, self.c2, image=self.img_tk)

    def draw_line(self, event):
        if str(event.type) == 'ButtonPress':
            self.Pointer.start = (event.x, event.y)
            self.Pointer.end = None
        elif str(event.type) == 'ButtonRelease':
            self.Pointer.end = (event.x, event.y)

        if self.Pointer.start is not None:
            tmp = normalize(copy.copy(self.workspace))
            cv2.line(tmp, tuple(self.Pointer.start), tuple((event.x, event.y)), (255, 255, 255), thickness=1,
                     lineType=8)
            self.tk_show(tmp)

        if self.Pointer.check():
            self.set_params()
            self.show(highlight_border=True, color=(0, 0, 0))

    def cancel(self, show_img=True):
        self.Pointer.reset()
        self.params = None
        if show_img:
            self.tk_show(normalize(self.workspace))

    def _update(self, input):
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
            input = foreground * alpha_map + input * (1 - alpha_map)

        return input

    def draw_on_canvas(self):
        if self.params is None:
            return
        self.workspace = self._update(input=self.workspace)
        self.canvas = self._update(input=self.canvas)

        # Show the updated workspace and Canvas
        self.tk_show(normalize(self.workspace))

        if torch.is_tensor(self.params):
            self.S = torch.cat((self.S, self.params.cpu()), dim=1)
        else:
            self.S = torch.cat((self.S, torch.tensor(self.params)), dim=1)
        self.cancel(show_img=False)

    def change_thickness(self, event):
        self.set_params()
        self.show()

    def preprocess(self):
        img = torch.tensor(cv2.resize(self.img, (256, 256)) / 255).permute(2, 0, 1).unsqueeze(0).float()
        canvas = torch.tensor(cv2.resize(self.canvas, (256, 256))).permute(2, 0, 1).unsqueeze(0).float()

        s_c = self.S[:, -self.ctx:, :].float()
        s_t = torch.rand_like(s_c)
        return dict(img=img, canvas=canvas, strokes_ctx=s_c, strokes_seq=s_t)

    def model_prediction(self):
        batch = self.preprocess()
        with torch.no_grad():
            self.params = self.model.generate(batch)["fake_data_random"]
        self.show(highlight_border=True, color=(1, 0, 0))


def run(args):
    root = Tk()
    win = WindowApp(root, args)
    root.mainloop()


if __name__ == '__main__':
    # Extra parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, help='path containing the reference image to be painted')
    parser.add_argument("--painter_config", type=str, help='configuration file of the renderer')
    parser.add_argument("--checkpoint", type=str, help='path the pth.tar model')
    args = parser.parse_args()

    # config of the window
    BG_COLOR = '#c8c7c9' # '24272b'
    SIZE = [514, 750]

    # gui
    run(args)
