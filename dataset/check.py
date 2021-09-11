import os
import shutil

if __name__ == '__main__':

    path = '/data/eperuzzo/decomposition_output_v1/'

    files = os.listdir(path)


    for file in files:
        assert file.startswith('ADE')

    print(f'Number of folders: {len(files)}')

    for file in files:

        tmp = os.listdir(os.path.join(path, file))

        if len(tmp) == 0:
            shutil.rmtree(os.path.join(path, file))
            print(f'file {file} is empty')