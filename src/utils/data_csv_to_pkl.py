import pickle
import numpy as np
import argparse
import pandas as pd
import os

def data_csv_to_pkl(dataset_path, data_nickname):
    data_df = pd.read_csv(dataset_path, encoding='unicode-escape', index_col=0)
    mean_df = data_df.mean()
    std_df = data_df.std()
    mean = mean_df.values.astype('float32')
    std = std_df.values.astype('float32')
    mean_and_std = (mean, std)

    mask_df = 1 - data_df.isnull()
    mask = mask_df.values.astype('float32')
    data_df = data_df.fillna(0)
    data = data_df.values
    data = data.astype('float32')
    data_and_mask = (data, mask)


    os.makedirs(f'data/{data_nickname}', exist_ok=True)
    with open(f'data/{data_nickname}/data.pkl', 'wb') as f:
        pickle.dump(data_and_mask, f)
    with open(f'data/{data_nickname}/meanstd.pkl', 'wb') as f:
        pickle.dump(mean_and_std, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help='Local path to the dataset csv file')
    parser.add_argument('--data_nickname', type=str, required=True, help='Nickname for the dataset')
    args = parser.parse_args()
    data_csv_to_pkl(args.dataset_path, args.data_nickname)