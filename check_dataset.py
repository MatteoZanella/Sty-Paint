import os
import glob
import numpy as np
import PIL.Image as Image

if __name__ == '__main__':

    base_path = '/data/eperuzzo/ade_generative_brushstrokes_dataset/'

    n_images = os.listdir(base_path)
    print(f'Number of images : {len(n_images)}')
    for img in n_images:
        assert img.startswith('ADE_train')

        files = os.listdir(os.path.join(base_path, img))

        assert len(files) == 4

        for file in files:
            if not file.startswith('render'):
                assert os.path.getsize(os.path.join(base_path, img, file)) > 0

        strokes = np.load(os.path.join(base_path, img, 'strokes_params.npz'))['x_ctt']
        renders = glob.glob(os.path.join(base_path, img, 'render_lkh_col2_area1_pos0_cl7/*.jpg'))

        assert len(renders) == strokes.shape[1]
        #print(f'Number of renders: {len(renders)}')

        img_ref = Image.open(os.path.join(os.path.join(base_path, img, img + '.jpg')))
        if img_ref.mode != 'RGB':
            print(img)