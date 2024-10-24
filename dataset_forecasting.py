import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch

class Forecasting_Dataset(Dataset):
    def __init__(self, datatype, mode="train", time_weaver=False):
        self.history_length = 168
        self.pred_length = 24
        self.time_weaver = time_weaver

        if datatype == 'electricity':
            datafolder = './data/electricity_nips'
            self.test_length= 24*7
            self.valid_length = 24*5
            
        self.seq_length = self.history_length + self.pred_length
            
        paths=datafolder+'/data.pkl' 
        #shape: (T x N)
        #mask_data is usually filled by 1
        with open(paths, 'rb') as f:
            self.main_data, self.mask_data = pickle.load(f)
        paths=datafolder+'/meanstd.pkl'
        with open(paths, 'rb') as f:
            self.mean_data, self.std_data = pickle.load(f)
            
        self.main_data = (self.main_data - self.mean_data) / self.std_data


        total_length = len(self.main_data)

        # if time_weaver:
        # categorical metadata for Time Weaver
        date_vector = pd.date_range(start='1/1/2011', periods=total_length, freq='H')
        month_vector = date_vector.month
        day_vector = date_vector.day
        date_meta = np.stack([month_vector, day_vector], axis=1)
        # could be more efficient and more robust to weird choices for categorical metadata - like years

        for i in range(date_meta.shape[1]):
            date_meta_1hot = np.zeros((date_meta.shape[0], np.max(date_meta[:,i])))
            date_meta_1hot[np.arange(date_meta.shape[0]), date_meta[:,i]-1] = 1

            if i == 0:
                date_meta_1hot_all = date_meta_1hot
            else:
                date_meta_1hot_all = np.concatenate((date_meta_1hot_all, date_meta_1hot), axis=1)
        
        # User data doesn't help for this experiment
        # user_meta = np.ones(self.main_data.shape)
        # self.cat_meta = np.concatenate([date_meta_1hot_all, user_meta], axis=1)
        self.cat_meta = date_meta_1hot_all

        # interleave the blocks of t/v/t data for Time Weaver with an 80/10/10 split     
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


    def __getitem__(self, orgindex):
        index = self.use_index[orgindex]
        # Time Weaver: change mask so that only one of the features is to be predicted
        target_mask = self.mask_data[index:index+self.seq_length].copy()
        target_mask[-self.pred_length:] = 0. #pred mask for test pattern strategy
        s = {
            'observed_data': self.main_data[index:index+self.seq_length],
            'observed_mask': self.mask_data[index:index+self.seq_length],
            'gt_mask': target_mask,
            'timepoints': np.arange(self.seq_length) * 1.0, 
            'feature_id': np.arange(self.main_data.shape[1]) * 1.0, 
            'cat_meta': self.cat_meta[index:index+self.seq_length]
        }

        return s
    def __len__(self):
        return len(self.use_index)

def get_dataloader(datatype, device, batch_size=8, time_weaver=False):
    dataset = Forecasting_Dataset(datatype,mode='train', time_weaver=time_weaver)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = Forecasting_Dataset(datatype,mode='valid', time_weaver=time_weaver)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = Forecasting_Dataset(datatype,mode='test', time_weaver=time_weaver)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=0)

    scaler = torch.from_numpy(dataset.std_data).to(device).float()
    mean_scaler = torch.from_numpy(dataset.mean_data).to(device).float()

    return train_loader, valid_loader, test_loader, scaler, mean_scaler