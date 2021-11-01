import pickle
import os
import shutil


if __name__ == '__main__':
    file = ''
    with open(file, 'rb') as f:
        to_move = pickle.load(f)


    source = ''
    destination = ''
    for name in to_move:
        src = os.path.join(source, name)
        dst = os.path.join(destination, name)
        shutil.copytree(src, dst)