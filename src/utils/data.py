import pickle
from torch.utils.data import DataLoader, Dataset
import yaml
import numpy as np
import torch
from sklearn.decomposition import PCA
import os
import math

from src.utils.utils import data_csv_to_pkl, gen_mask

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
        training_feature_sample_size,
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
        self.training_feature_sample_size = training_feature_sample_size
        self.train = train

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

        # finding gap sizes
        n_gap_slices = T - self.n_history_slices
        n_gaps = self.n_history_slices // self.history_block_size
        history_block_gap = math.ceil(n_gap_slices / n_gaps)


        # finding features retained idx
        if self.feature_retention_strategy == 'pca loadings':
            # reverse order
            self.condit_feature_idx = pca_loading_selection(self.main_data, self.n_feature_slices)[::-1]
        else:
            raise NotImplementedError
            
        mask_metadata = {
            'condit_feature_idx': self.condit_feature_idx, 
            'history_block_size': self.history_block_size,
            'history_block_gap': history_block_gap}
        with open(f"{save_folder}/presence_mask_metadata.yaml", 'w') as f:
            # conditional features stored in decreasing order of importance according to PCA
            yaml.dump(mask_metadata, f)

        total_length = len(self.main_data)
        
        if self.train:
            all_index = list(range(total_length-self.window_length))     
            self.use_index = all_index
        else:
            non_overlapping_index = list(range(0, total_length-self.window_length, self.window_length))
            self.use_index = non_overlapping_index


    def __getitem__(self, orgindex):

        t_index = self.use_index[orgindex]

        if self.train and self.main_data.shape[1] > self.training_feature_sample_size:
            # sample features
            rand_feature_idxs = np.random.choice(self.main_data.shape[1], self.training_feature_sample_size, replace=False)
            s = {
                'observed_data': self.main_data[t_index:t_index+self.window_length][:, rand_feature_idxs], # will be db query in future
                'presence_mask': gen_mask(
                    features_in_window = rand_feature_idxs,
                    start_idx = t_index,
                    end_idx = t_index+self.window_length,
                    save_folder = self.save_folder
                    ),
                'timepoints': np.arange(self.window_length, dtype=np.int32), 
                'feature_id': rand_feature_idxs, 
            }
        else:
            # all features
            s = {
                'observed_data': self.main_data[t_index:t_index+self.window_length], # will be db query in future
                'presence_mask': gen_mask(
                    features_in_window = np.arange(self.main_data.shape[1]),
                    start_idx = t_index,
                    end_idx = t_index+self.window_length,
                    save_folder = self.save_folder
                    ),
                'timepoints': np.arange(self.window_length, dtype=np.int32), 
                'feature_id': np.arange(self.main_data.shape[1], dtype=np.int32), 
            }
            print(s['presence_mask'].sum(), flush=True)

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
    training_feature_sample_size,
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
        training_feature_sample_size = training_feature_sample_size,
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
        training_feature_sample_size = training_feature_sample_size,
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
    pca_sorted_features = np.argsort(np.abs(pca.components_))[0].tolist()
    return pca_sorted_features[-n_features:]
