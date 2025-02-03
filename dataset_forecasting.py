import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
from sklearn.decomposition import PCA

class Forecasting_Dataset(Dataset):
    def __init__(
            self,
            datatype,
            mode="train",
            time_weaver=False,
            true_unconditional=False,
            history_length=168,
            ):
        if not true_unconditional:
            self.history_length = history_length
        else:
            self.history_length = 0
        self.pred_length = 24
        self.time_weaver = time_weaver

        # will be instantiated if feature selection is used
        self.n_condit_features = None
        self.condit_features = None
        self.pred_var = None
        self.pred_var_idx = None

        if datatype == 'electricity':
            datafolder = './data/electricity_nips'
            
        self.seq_length = self.history_length + self.pred_length
            
        paths=datafolder+'/data.pkl' 
        # shape: (T x N)
        # mask_data is usually filled by 1
        with open(paths, 'rb') as f:
            self.main_data, true_presence_mask = pickle.load(f)
        paths=datafolder+'/meanstd.pkl'
        with open(paths, 'rb') as f:
            self.mean_data, self.std_data = pickle.load(f)

        self.main_data = (self.main_data - self.mean_data) / self.std_data
        self.true_presence_mask = true_presence_mask

        self.mask_data = true_presence_mask
            # true presence mask and mask data are the same
        total_length = len(self.main_data)

        # Whenever we expand to other datasets and there might be metadata to use in Time Weaver:
        if time_weaver:
            meta_path = datafolder + '/metadata.pkl'
            with open(meta_path, 'rb') as f:
                meta_data = pickle.load(f)
            # categorical metadata for Time Weaver
            for i in range(meta_data.shape[1]): # iterating over feature index
                meta_1hot = np.zeros((meta_data.shape[0], np.max(meta_data[:,i])))
                meta_1hot[np.arange(meta_data.shape[0]), meta_data[:,i]-1] = 1
                if i == 0:
                    meta_1hot_all = meta_1hot
                else:
                    meta_1hot_all = np.concatenate((meta_1hot_all, meta_1hot), axis=1)

            self.metadata = meta_1hot_all

        # interleave the blocks of t/v/t data with an 80/10/10 split     
        all_index = list(range(total_length-self.seq_length))
        self.use_index = []
        step_idx = 0
        while step_idx < total_length-self.seq_length:
            if mode == 'train': 
                top_train_idx = min(8, total_length-step_idx)
                self.use_index += all_index[step_idx: step_idx + top_train_idx]
                step_idx += 7 + self.pred_length*3
            elif mode == 'valid': #valid
                if step_idx == 0:
                    step_idx = 7 + self.pred_length
                else:
                    self.use_index.append(all_index[step_idx])
                    step_idx += 7 + self.pred_length*3
            elif mode == 'test': #test
                if step_idx == 0:
                    step_idx = 7 + self.pred_length + self.pred_length
                else:
                    self.use_index.append(all_index[step_idx])
                    step_idx += 7 + self.pred_length*3
            else:
                raise ValueError('Invalid mode')
        # if training over all data
        if mode == 'train':
            self.use_index = all_index


    def __getitem__(self, orgindex):
        """
        If condit features are used, (ex. idx 2,3)
        >>> gt_mask = [1, 1, 1, 1]
        ...           [1, 1, 1, 1]
        ...           ------------ <- pred start
        ...           [1, 1, 1, 0]
        >>> observed_mask = [1, 1, 1, 1]
        ...                 [1, 1, 1, 1]
        ...                 ------------ <- pred start
        ...                 [1, 1, 1, 1]   
        """
        index = self.use_index[orgindex]
        target_mask = self.mask_data[index:index+self.seq_length].copy()
        if self.pred_var is not None:
            target_mask[-self.pred_length:, self.pred_var_idx] = 0. # zero after prediction start for the target variable
        else:
            target_mask[-self.pred_length:, :] = 0. # zero after prediction start for all variables

        mask_data = self.mask_data[index:index+self.seq_length].copy()
        s = {
            'observed_data': self.main_data[index:index+self.seq_length],
            'observed_mask': mask_data,
            'gt_mask': target_mask,
            'timepoints': np.arange(self.seq_length) * 1.0, 
            'feature_id': np.arange(self.main_data.shape[1]) * 1.0, 
        }
        if self.time_weaver:
            s['metadata'] = self.metadata[index:index+self.seq_length]
        return s
    
    def __len__(self):
        return len(self.use_index)


def get_dataloader(datatype, device, batch_size=8, time_weaver=False, true_unconditional=False, history_length=168, n_condit_features=-1, condit_strat="pca", config=None):
    dataset = Forecasting_Dataset(datatype,
        mode='train',
        time_weaver=time_weaver,
        true_unconditional=true_unconditional,
        history_length=history_length,
        )
    valid_dataset = Forecasting_Dataset(datatype,
        mode='valid',
        time_weaver=time_weaver,
        true_unconditional=true_unconditional,
        history_length=history_length,
        )
    test_dataset = Forecasting_Dataset(datatype,
        mode='test',
        time_weaver=time_weaver,
        true_unconditional=true_unconditional,
        history_length=history_length,
        )

    scaler = torch.from_numpy(dataset.std_data).to(device).float()
    mean_scaler = torch.from_numpy(dataset.mean_data).to(device).float()

    if n_condit_features > 0:
        if config['model']['condit_features'] is not None:
            condit_features = np.array(config['model']['condit_features'])
            pred_var = np.array(config['model']['pred_var'])
            pred_var_idx = np.array(config['model']['pred_var_idx'])
        else:
            if condit_strat == "pca":
                condit_features = pca_feature_selection(dataset.main_data, n_condit_features)
            elif condit_strat == "cosine":
                condit_features = cosine_feature_selection(dataset.main_data, n_condit_features)
            elif condit_strat == "random":
                condit_features = np.random.choice(dataset.main_data.shape[1], n_condit_features, replace=False)

            pred_var = np.random.choice(condit_features, 1)                  # random for experiments, but should be customizable in the future
            pred_var_idx = np.where(condit_features == pred_var)[0][0]

        for ds in [dataset, valid_dataset, test_dataset]:
            ds.n_condit_features = n_condit_features
            ds.condit_features = condit_features
            ds.pred_var = pred_var
            ds.pred_var_idx = pred_var_idx
            ds.main_data = ds.main_data[:, ds.condit_features]
            ds.mask_data = ds.mask_data[:, ds.condit_features]

        scaler = scaler[condit_features]
        mean_scaler = mean_scaler[condit_features]

    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=1)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=0)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=0)

    return train_loader, valid_loader, test_loader, scaler, mean_scaler



def pca_feature_selection(data, n_condit_features):
    print('\n$$$ PCA $$$\n')
    pca = PCA(n_components=1)
    pca.fit(data)
    pca_sorted_features = np.argsort(np.abs(pca.components_))
    return pca_sorted_features[0, -n_condit_features:]

def cosine_feature_selection(data, n_condit_features):
    print('\n$$$ Cosine $$$\n')
    cosine_sim = np.zeros((data.shape[1], data.shape[1]))
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            cosine_sim[i,j] = np.dot(data[:,i], data[:,j]) / (np.linalg.norm(data[:,i]) * np.linalg.norm(data[:,j]) + 1e-6)
    cosine_sorted_features = np.argsort(np.sum(np.abs(cosine_sim), axis=0))
    return cosine_sorted_features[-n_condit_features:]