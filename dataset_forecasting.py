import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch

class Forecasting_Dataset(Dataset):
    def __init__(self, datatype, mode="train", time_weaver=False, true_unconditional=False):
        if not true_unconditional:
            self.history_length = 168
        else:
            self.history_length = 0
        self.pred_length = 24
        self.time_weaver = time_weaver

        if datatype == 'electricity':
            datafolder = './data/electricity_nips'
            
        self.seq_length = self.history_length + self.pred_length
            
        paths=datafolder+'/data.pkl' 
        # shape: (T x N)
        # mask_data is usually filled by 1
        with open(paths, 'rb') as f:
            self.main_data, self.mask_data = pickle.load(f)
        paths=datafolder+'/meanstd.pkl'
        with open(paths, 'rb') as f:
            self.mean_data, self.std_data = pickle.load(f)
            
        self.main_data = (self.main_data - self.mean_data) / self.std_data


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
        index = self.use_index[orgindex]
        target_mask = self.mask_data[index:index+self.seq_length].copy()
        target_mask[-self.pred_length:] = 0. #pred mask for test pattern strategy
        s = {
            'observed_data': self.main_data[index:index+self.seq_length],
            'observed_mask': self.mask_data[index:index+self.seq_length],
            'gt_mask': target_mask,
            'timepoints': np.arange(self.seq_length) * 1.0, 
            'feature_id': np.arange(self.main_data.shape[1]) * 1.0, 
        }
        if self.time_weaver:
            s['metadata'] = self.metadata[index:index+self.seq_length]

        return s
    def __len__(self):
        return len(self.use_index)

def get_dataloader(datatype, device, batch_size=8, time_weaver=False, true_unconditional=False):
    dataset = Forecasting_Dataset(datatype,mode='train', time_weaver=time_weaver, true_unconditional=true_unconditional)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = Forecasting_Dataset(datatype,mode='valid', time_weaver=time_weaver, true_unconditional=true_unconditional)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = Forecasting_Dataset(datatype,mode='test', time_weaver=time_weaver, true_unconditional=true_unconditional)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=0)

    scaler = torch.from_numpy(dataset.std_data).to(device).float()
    mean_scaler = torch.from_numpy(dataset.mean_data).to(device).float()

    return train_loader, valid_loader, test_loader, scaler, mean_scaler