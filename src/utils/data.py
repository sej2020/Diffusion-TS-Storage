import pickle
from torch.utils.data import DataLoader, Dataset
import yaml
import numpy as np
import torch
from sklearn.decomposition import PCA
import os
import math

from src.utils.utils import data_csv_to_pkl

class ConditionalDataset(Dataset):
    def __init__(
        self,
        dataset,
        device,
        save_folder,
        compression,
        feature_retention_strategy,
        history_block_size,
        model_param_proportion,
        history_to_feature_ratio,
        window_length,
        data_dayfirst = False,
        train = True,
    ):
        self.dataset = dataset
        self.device = device
        self.save_folder = save_folder
        self.compression = compression
        self.feature_retention_strategy = feature_retention_strategy
        self.history_block_size = history_block_size
        self.model_param_proportion = model_param_proportion
        self.history_to_feature_ratio = history_to_feature_ratio
        self.window_length = window_length

        if not os.path.exists(f'data/{dataset}/data.pkl'):
            data_csv_to_pkl(f"data/{dataset}.csv", f"data/{dataset}", dayfirst=data_dayfirst)
        
        with open(f"data/{dataset}/data.pkl", 'rb') as f:
            self.main_data = pickle.load(f)
        with open(f"data/{dataset}/meanstd.pkl", 'rb') as f:
            self.mean_data, self.std_data = pickle.load(f)

        # normalize the data
        self.main_data = (self.main_data - self.mean_data) / (self.std_data + 1e-9)

        # We have a dataset with T time points and N features (TxN data matrix).
        T, N = self.main_data.shape
        n_total_points = T * N
        n_compressed_points = int(n_total_points * self.compression)
        # We want to keep a certain proportion of the data as model parameters: n_model_points = int(n_compressed_points * model_param_proportion)
        self.n_model_points = 616033 # but for right now, it is just the base model, which has about 600k parameters
        assert self.n_model_points < n_compressed_points, "Model parameters are more than the compressed data, please make your model smaller"
        exp_n_condit_points = n_compressed_points - self.n_model_points
        exp_point_compression = exp_n_condit_points / n_total_points

        # sets number of history slices and features retained
        self._calc_slice_counts(exp_point_compression)

        # finding history slices idx
        n_history_blocks = self.n_history_slices // self.history_block_size
        block_offset = math.ceil(T / n_history_blocks)
        self.history_slice_idx = []
        for i in range(0, T, block_offset):
            self.history_slice_idx += [i + j for j in range(self.history_block_size) if i + j < T]
        remaining_slices = self.n_history_slices - len(self.history_slice_idx)

        for i in range(0, remaining_slices):
            self.history_slice_idx.append(i*block_offset + self.history_block_size)

        # finding features retained idx
        if self.feature_retention_strategy == 'pca loadings':
            self.feature_idx = pca_loading_selection(self.main_data, self.n_feature_slices)
        else:
            raise NotImplementedError
            
        # construct the mask
        mask_data = np.zeros_like(self.main_data)
        mask_data[self.history_slice_idx] = 1
        mask_data[:, self.feature_idx] = 1
        self.mask_data = mask_data

        with open(f"{save_folder}/presence_mask.pkl", 'wb') as f:
            pickle.dump(self.mask_data, f)

        total_length = len(self.main_data)
        
        if train:
            all_index = list(range(total_length-self.window_length))     
            self.use_index = all_index
        else:
            non_overlapping_index = list(range(0, total_length-self.window_length, self.window_length))
            self.use_index = non_overlapping_index


    def __getitem__(self, orgindex):
        index = self.use_index[orgindex]

        s = {
            'observed_data': self.main_data[index:index+self.window_length],
            'presence_mask': self.mask_data[index:index+self.window_length],
            'timepoints': np.arange(self.window_length, dtype=np.int32), 
            'feature_id': np.arange(self.main_data.shape[1], dtype=np.int32), 
        }
        return s
    

    def __len__(self):
        return len(self.use_index)


    def _calc_slice_counts(self, point_compression):
        T = self.main_data.shape[0]
        N = self.main_data.shape[1]
        z = self.history_to_feature_ratio
        C = point_compression

        # no intuition here, you'll have to do the algebra
        self.n_feature_slices = math.floor((N + N*z - math.sqrt(N**2*(1 + 2*z - 4*C*z + z**2)))/(2*z))
        self.n_history_slices = math.floor((T*(N + N*z - math.sqrt(N**2*(1 + 2*z - 4*C*z + z**2))))/(2*N))

        # verify that the math is correct
        self.n_condit_points = int(self.n_history_slices*N + self.n_feature_slices*T - self.n_history_slices*self.n_feature_slices)
        print(f"Expected points retained: {int(T*N*C)},\nActual: {self.n_condit_points}")
        if not math.isclose(self.n_condit_points, T*N*C, rel_tol=.01):
            actual_compression = (self.n_condit_points + self.n_model_points) / (T*N)
            print("Warning compression is off by more than 1%")
            print(f"Expected compression: {self.compression},\nActual: {round(actual_compression, 4)}")


def get_dataloader(
    dataset,
    device,
    save_folder,
    compression,
    feature_retention_strategy,
    history_block_size,
    model_param_proportion,
    history_to_feature_ratio,
    window_length,
    data_dayfirst = False,
    ):

    conditional_dataset_train = ConditionalDataset(
        dataset = dataset,
        device = device,
        save_folder = save_folder,
        compression = compression,
        feature_retention_strategy = feature_retention_strategy,
        history_block_size = history_block_size,
        model_param_proportion = model_param_proportion,
        history_to_feature_ratio = history_to_feature_ratio,
        window_length = window_length,
        data_dayfirst = data_dayfirst,
        )
    
    conditional_dataset_eval = ConditionalDataset(
        dataset = dataset,
        device = device,
        save_folder = save_folder,
        compression = compression,
        feature_retention_strategy = feature_retention_strategy,
        history_block_size = history_block_size,
        model_param_proportion = model_param_proportion,
        history_to_feature_ratio = history_to_feature_ratio,
        window_length = window_length,
        data_dayfirst = data_dayfirst,
        train = False,
        )
    
    scaler = torch.from_numpy(conditional_dataset_train.std_data).to(device).float()
    mean_scaler = torch.from_numpy(conditional_dataset_train.mean_data).to(device).float()

    with open(f"config/train_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    train_loader = DataLoader(
        conditional_dataset_train, batch_size=config['train']['batch_size'], shuffle=1)

    eval_loader = DataLoader(
        conditional_dataset_eval, batch_size=1, shuffle=0)

    return train_loader, eval_loader, scaler, mean_scaler



def pca_loading_selection(data, n_features):
    print('\n$$$ PCA $$$\n')
    pca = PCA(n_components=1)
    pca.fit(data)
    pca_sorted_features = np.argsort(np.abs(pca.components_))
    return list(pca_sorted_features[0, -n_features:])


