import pandas as pd
import glob
import os
import argparse
import numpy as np


def get_args():
    # settings
    parser = argparse.ArgumentParser(description='DATASET SPLITTING')
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_path', required=True)
    return parser.parse_args()



if __name__ == '__main__':

    args = get_args()

    annotations = glob.glob(os.path.join(args.data_path, 'annotations/training/*.png'))
    images = glob.glob(os.path.join(args.data_path, 'images/training/*.jpg'))

    # Create the dataframe
    df = pd.DataFrame(list(zip(sorted(annotations), sorted(images))), columns=['Annotations', 'Images'])

    # Chunk the dataframe
    df_chunks = np.array_split(df, 5)

    # Save each chunk
    for i in range(len(df_chunks)):
        df_chunks[i].to_csv(os.path.join(args.output_path, f'chunk_{i}.csv'))