import pickle
from torch.utils.data import DataLoader, Dataset
import yaml
import numpy as np
import torch
from sklearn.decomposition import PCA
import os
import math
import concurrent.futures

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
        data_to_model_ratio,
        history_to_feature_ratio,
        window_length,
        training_feature_sample_size,
        data_dayfirst = False,
        train = True,
        **kwargs
    ):
        self.dataset = dataset
        self.device = device
        self.save_folder = save_folder
        self.compression = compression
        self.feature_retention_strategy = feature_retention_strategy
        self.history_block_size = history_block_size
        self.data_to_model_ratio = data_to_model_ratio
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
        data_point_percnt = data_to_model_ratio / (data_to_model_ratio + 1)

        exp_n_condit_points = int(n_compressed_points * data_point_percnt)
        exp_point_compression = exp_n_condit_points / n_total_points

        # sets number of history slices and features retained
        self._calc_slice_counts(exp_point_compression)

        # finding gap sizes
        n_gap_slices = T - self.n_history_slices
        n_gaps = self.n_history_slices // self.history_block_size # should be the same as number of history blocks
        history_block_gap = math.ceil(n_gap_slices / n_gaps) # conservative size (bigger than strictly necessary)

        # finding features retained idx
        most_useful = pca_sorted_features(self.main_data)
        if self.feature_retention_strategy == 'pca':
            self.condit_feature_idx = most_useful[:self.n_feature_slices]
        elif self.feature_retention_strategy == 'select':

            if 'selected_features' not in kwargs:
                raise ValueError("Please provide selected features to retain")

            with open(f'data/{dataset}/labels.pkl', 'rb') as f:
                # two dictionaries: {'start': int, 'interval': int} and {str: int}
                (_, var_names) = pickle.load(f)
            
            # if len(selected_features) > self.n_feature_slices, then it will be truncated to n_feature_slices
            selected_feature_idxs = [var_names[v] for v in kwargs['selected_features']][:self.n_feature_slices]
            
            # but if not, we fill the rest with the most useful features
            useful_idx = 0
            selected_pca_features = []
            while len(selected_feature_idxs) + len(selected_pca_features) < self.n_feature_slices:
                if most_useful[useful_idx] not in selected_feature_idxs:
                    selected_pca_features.append(most_useful[useful_idx])
                useful_idx += 1
            self.condit_feature_idx = selected_pca_features + selected_feature_idxs # need pca features to show up first for querying purposes
            
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


    async def _load_data(self, start_idx, end_idx, feature_idxs):
        # This function is a placeholder for loading data from a database
        # In the future, this function should be implemented to load data asynchronously
        pass


    def __getitem__(self, orgindex):

        # right now: loads whole dataset into memory and indexs into it
        # in the future: load data from db and prefetch next index

        t_index = self.use_index[orgindex]

        if self.train and self.main_data.shape[1] > self.training_feature_sample_size:
            # sample features
            # if self.prefetched_data index matches t_index, use it
            # if self.prefetched_data.done() and self.prefetched_data.index == t_index:
            #     s = self.prefetched_data.result()
            #     self.prefetched_data = None
            # else, load data from db
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
            # executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            # next_rand_feature_idxs = np.random.choice(self.main_data.shape[1], self.training_feature_sample_size, replace=False)
            # self.prefetched_data = executor.submit(self._load_data, t_index+1, t_index+self.window_length+1, next_rand_feature_idxs)

        else:
            # all features
            # if self.prefetched_data index matches t_index, use it
            # else, load data from db
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
            # executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            # self.prefetched_data = executor.submit(self._load_data, t_index+self.window_length, t_index+self.window_length*2, np.arange(self.main_data.shape[1]))

        return s
    

    def __len__(self):
        return len(self.use_index)


    def _calc_slice_counts(self, point_compression: float):
        T = self.main_data.shape[0]
        N = self.main_data.shape[1]
        z = self.history_to_feature_ratio
        C = point_compression

        # no intuition here, you'll have to do the algebra
        self.n_feature_slices = math.floor((N + N*z - math.sqrt(N**2*(1 + 2*z - 4*C*z + z**2)))/(2*z))
        self.n_history_slices = math.floor((T*(N + N*z - math.sqrt(N**2*(1 + 2*z - 4*C*z + z**2))))/(2*N))

        # verify that the math is correct
        self.n_condit_points = int(self.n_history_slices*N + self.n_feature_slices*T - self.n_history_slices*self.n_feature_slices)
        print(f"Expected datapoints retained: {int(T*N*C)},\nActual: {self.n_condit_points}")


def get_dataloader(config: str | dict, save_folder: str) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    """
    Creates the dataloaders for the training and evaluation datasets.

    Args:
        config (str | dict): The configuration file or configuration dictionary containing the dataset parameters.
        save_folder (str): The folder where the data is to be saved.

    Returns:
        tuple: A tuple containing the training dataloader, evaluation dataloader, stdev tensor, and mean tensor.
    """

    if isinstance(config, str):
        config_name = config
        with open(config_name, 'r') as f:
            config = yaml.safe_load(f)

    conditional_dataset_train = ConditionalDataset(
        dataset = config['data']['dataset'],
        device = config['train']['device'],
        save_folder = save_folder,
        compression = config['compression']['compression_rate'],
        feature_retention_strategy = config['compression']['feature_retention_strategy'],
        history_block_size = config['compression']['history_block_size'],
        data_to_model_ratio = config['compression']['data_to_model_ratio'],
        history_to_feature_ratio = config['compression']['history_to_feature_ratio'],
        window_length = config['train']['window_length'],
        training_feature_sample_size = config['model']['training_feature_sample_size'],
        data_dayfirst = config['data']['day_first'],
        selected_features = config['compression']['selected_features'] if config['compression']['feature_retention_strategy'] == 'select' else None,
        )
    
    conditional_dataset_eval = ConditionalDataset(
        dataset = config['data']['dataset'],
        device = config['train']['device'],
        save_folder = save_folder,
        compression = config['compression']['compression_rate'],
        feature_retention_strategy = config['compression']['feature_retention_strategy'],
        history_block_size = config['compression']['history_block_size'],
        data_to_model_ratio = config['compression']['data_to_model_ratio'],
        history_to_feature_ratio = config['compression']['history_to_feature_ratio'],
        window_length = config['train']['window_length'],
        training_feature_sample_size = config['model']['training_feature_sample_size'],
        data_dayfirst = config['data']['day_first'],
        train = False,
        selected_features = config['compression']['selected_features'] if config['compression']['feature_retention_strategy'] == 'select' else None,
        )
    
    scaler = torch.from_numpy(conditional_dataset_train.std_data).to(config['train']['device']).float()
    mean_scaler = torch.from_numpy(conditional_dataset_train.mean_data).to(config['train']['device']).float()


    train_loader = DataLoader(
        conditional_dataset_train, batch_size=config['train']['batch_size'], shuffle=1
        )

    eval_loader = DataLoader(
        conditional_dataset_eval, batch_size=2, shuffle=0
        )

    return train_loader, eval_loader, scaler, mean_scaler



def pca_sorted_features(data: np.ndarray) -> list:
    """
    Runs PCA on the data and returns the indices of the features sorted by their absolute loadings. This roughly
    corresponds to the importance of the features in the PCA model. The features are sorted from most important to least important.

    Args:
        data (np.ndarray): The data to run PCA on. Should be a 2D array with shape (n_samples, n_features).
    
    Returns:
        list: A list of indices of the features sorted by their absolute loadings.
    """
    pca = PCA(n_components=1)
    pca.fit(data)
    sorted_features = np.argsort(np.abs(pca.components_))[0].tolist()
    return sorted_features[::-1] # best to worst
