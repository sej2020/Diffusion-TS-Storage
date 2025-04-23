import pickle
import math
import pandas as pd
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import os
import yaml
from src.model.main_model import CSDI
import warnings
from fvcore.nn import FlopCountAnalysis


def adjust_model_architecture(config: dict, dataset_len: int, dataset_dim: int) -> dict:
    """
    Calculates the number of parameters permitted for the model based on the compression rate and data to model ratio.
    If the requested number of parameters is greater than the permitted number, the model architecture is adjusted to fit within the limits.
    If the requested number of parameters is less than the permitted number, the data to model ratio is adjusted to fit within the limits.

    Args:
        config (dict): the configuration dictionary
        dataset_len (int): the number of timesteps in the dataset
        dataset_dim (int): the number of features in the dataset

    Returns:
        config (dict): the updated configuration dictionary
    """
    dummy_model = CSDI(config, dataset_dim, device=config['train']['device'])
    requested_params = sum(p.numel() for p in dummy_model.parameters() if p.requires_grad)
    del dummy_model

    total_datapoints = dataset_len * dataset_dim
    comp_rate = config['compression']['compression_rate']
    data_to_model_ratio = config['compression']['data_to_model_ratio']

    data_point_percnt = data_to_model_ratio / (data_to_model_ratio + 1)
    n_params_allowed = int(comp_rate * total_datapoints * (1 - data_point_percnt))
    
    assert n_params_allowed > 50_000, f"The model must have at least 50000 parameters (approx.). With the current config, only {n_params_allowed} parameters can be allocated to the model. Please adjust the compression rate up or data to model ratio down."

    if n_params_allowed > requested_params:
        warnings.warn(f"You have requested a model with {requested_params} parameters. For your chosen compression rate of {comp_rate} and data to model ratio of {data_to_model_ratio}, \
you may have up to {n_params_allowed} parameters. Your data to model ratio will be increased accordingly.")
        total_allowed = comp_rate * total_datapoints
        n_datapoints_allowed = total_allowed - requested_params
        data_to_model_ratio = math.floor((n_datapoints_allowed / requested_params)*10)/10
        config['compression']['data_to_model_ratio'] = data_to_model_ratio
    else:
        new_requested_params = requested_params
        while new_requested_params > n_params_allowed:
            if config['diffusion']['channels'] / config['diffusion']['layers'] > 12: # try to maintain at least a 12:1 ratio
                config['diffusion']['channels'] -= 8
                config['diffusion']['nheads'] -= 1 # channels / nheads must be 8
            else:
                config['diffusion']['layers'] -= 1
            
            dummy_model = CSDI(config, dataset_dim, device=config['train']['device'])
            new_requested_params = sum(p.numel() for p in dummy_model.parameters() if p.requires_grad)
            del dummy_model

        warnings.warn(f"The model architecture specified in the config file has {requested_params} parameters, which is too large for a compression rate of {comp_rate} \
and a data to model ratio of {data_to_model_ratio}. The model architecture has been adjusted to have {new_requested_params} parameters.")
        warnings.warn(f"The model now has {config['diffusion']['layers']} layers, {config['diffusion']['channels']} channels, and {config['diffusion']['nheads']} heads.")

    return config


def gen_mask(features_in_window: np.ndarray, start_idx: int, end_idx: int, save_folder: str) -> np.ndarray:
    """
    Generates a mask for the presence of conditional data in a window. The mask is a binary matrix where 1s represent the presence of data
    and 0s represent the absence of data. The mask is generated based on the retained features, and the repeating history pattern.

    Args:
        features_in_window (np.ndarray): the indices of the features in the window
        start_idx (int): the start index of the window
        end_idx (int): the end index of the window
        save_folder (str): the folder where the mask metadata is saved

    Returns:
        mask (np.ndarray): the mask for the presence of conditional data in the window
    """
    T = end_idx - start_idx
    N = len(features_in_window)
    mask = np.zeros((T, N))

    with open(f"{save_folder}/presence_mask_metadata.yaml", 'r') as f:
        mask_metadata = yaml.safe_load(f)
    condit_feature_idx = mask_metadata['condit_feature_idx']
    history_block_size = mask_metadata['history_block_size']
    history_block_gap = mask_metadata['history_block_gap']

    # mask is 1 if the feature is in the conditional feature set
    condit_feature_set = set(condit_feature_idx)
    for feat_idx, feat in enumerate(features_in_window):
        if feat in condit_feature_set:
            mask[:, feat_idx] = 1
    

    # block and gap pattern
    pattern = np.array([1] * history_block_size + [0] * history_block_gap)
    pattern = np.tile(pattern, T // (history_block_size + history_block_gap) + 2)
    
    # where in block and gap pattern window starts
    start_in_pattern = start_idx % (history_block_size + history_block_gap)
    pattern_slice = pattern[start_in_pattern:start_in_pattern+T]
    
    # where there are blocks in the window, make the mask 1
    pattern_1_idx = np.where(pattern_slice == 1)[0]
    mask[pattern_1_idx, :] = 1
    
    return mask



def data_csv_to_pkl(dataset_path: str, save_folder: str, dayfirst: bool = False) -> None:
    """
    Loads a dataset from a csv file, performs linear interpolation, and saves the data as a pickle file.
    Also saves the mean and standard deviation of the data as a pickle file.

    Args:
        dataset_path (str): the path to the dataset csv file
        save_folder (str): the folder where the data and mean/std will be saved
        dayfirst (bool): whether the day is the first element in the date string
    
    Returns:
        None
    """
    os.makedirs(save_folder, exist_ok=True)
    data_df = pd.read_csv(dataset_path, encoding='unicode-escape', index_col=0)
    
    # mean and std
    mean_df = data_df.mean()
    std_df = data_df.std()
    mean = mean_df.values.astype('float32')
    std = std_df.values.astype('float32')
    mean_and_std = (mean, std)
    with open(f'{save_folder}/meanstd.pkl', 'wb') as f:
        pickle.dump(mean_and_std, f)

    # interpolate missing values
    data_df = data_df.interpolate(method='linear', limit_direction='both', axis=0)
    
    # time and var dicts
    cols = data_df.columns.to_list()
    time_indices = pd.to_datetime(data_df.index, dayfirst=dayfirst)
    time_diffs = time_indices[1:] - time_indices[:-1]
    time_diff = time_diffs.value_counts().idxmax().total_seconds()

    start_time = time_indices[0].timestamp()

    var_dict = {var: i for i, var in enumerate(cols)}
    time_dict = {'start': start_time, 'interval': time_diff}

    os.makedirs(save_folder, exist_ok=True)
    with open(f"{save_folder}/labels.pkl", 'wb') as f:
        pickle.dump((time_dict, var_dict), f)
    
    # save data
    data = data_df.values
    data = data.astype('float32')

    with open(f'{save_folder}/data.pkl', 'wb') as f:
        pickle.dump(data, f)




def train(model, config, train_loader, save_folder):
    # optimization
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)

    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=config["lr"] * 0.1, max_lr=config["lr"], step_size_up=10, mode='triangular2'
    )

    total_forward_passes = 0

    best_training_losses = set((1e10, 1e11, 1e12))
    for epoch in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_num, train_batch in enumerate(it, start=1):
                observed_data = train_batch['observed_data']
                presence_mask = train_batch['presence_mask']
                feature_id = train_batch['feature_id']
                optimizer.zero_grad()

                if epoch == 1 and batch_num == 5:
                    flops = FlopCountAnalysis(model, (observed_data, presence_mask, feature_id)).total()

                loss = model(observed_data, presence_mask, feature_id, is_train=1)
                total_forward_passes += 1

                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_num,
                        "epoch": epoch,
                    }, refresh=False)            
            lr_scheduler.step()

        avg_loss /= len(train_loader)
        if avg_loss < min(best_training_losses):
            print(f"Saving best model at epoch {epoch}")
            torch.save(model.state_dict(), f"{save_folder}/model.pth")
        if any(avg_loss < loss for loss in best_training_losses):
            best_training_losses.add(avg_loss)
            best_training_losses.remove(max(best_training_losses))
            print(f"Best training loss: {avg_loss}")
        else:
            print("Early stopping")
            break
        print(best_training_losses)
    print(f"Epoch {epoch}: avg_loss {avg_loss}")

    # reporting metrics
    with open(f"{save_folder}/flops.yaml", "w") as f:
        yaml.dump({"flops_per_forward_pass": flops, "total_forward_passes": total_forward_passes}, f)


def evaluate(model, test_loader, scaler, save_folder):
    
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        normalized_mse_total = 0
        normalized_mae_total = 0
        evalpoints_total = 0

        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_num, test_batch in enumerate(it, start=1):
                observed_data = test_batch['observed_data'] # [B, L, K]
                presence_mask = test_batch['presence_mask'] # [B, L, K]
                feature_id = test_batch['feature_id'] # [B, K]

                samples = model.generate(observed_data, presence_mask, feature_id, gen_noise_magnitude=0).squeeze(1) # [B, L, K]

                target_mask = 1 - presence_mask # [B, L, K]
                target = observed_data # [B, L, K]
                # put target and target_mask to the same device as samples
                target = target.to(samples.device)
                target_mask = target_mask.to(samples.device)

                mse_current = (
                    ((samples - target) * target_mask) ** 2
                ) * ((scaler + 1e-9) ** 2)
                mae_current = (
                    torch.abs((samples - target) * target_mask) 
                ) * (scaler + 1e-9)
                normalized_mse_current = mse_current / ((scaler + 1e-9) ** 2)
                normalized_mae_current = mae_current / (scaler + 1e-9)

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                normalized_mse_total += normalized_mse_current.sum().item()
                normalized_mae_total += normalized_mae_current.sum().item()
                evalpoints_total += target_mask.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_num": batch_num,
                    },
                    refresh=True,
                )

    # reporting metrics
    RMSE = np.sqrt(mse_total / evalpoints_total)
    MAE = mae_total / evalpoints_total
    NRMSE = np.sqrt(normalized_mse_total / evalpoints_total)
    NMAE = normalized_mae_total / evalpoints_total

    print("RMSE:", RMSE)
    print("MAE:", MAE)
    print("Normalized RMSE:", NRMSE)
    print("Normalized MAE:", NMAE)

    with open(f"{save_folder}/result.yaml", "w") as f:
        yaml.dump({"RMSE": float(RMSE), "MAE": float(MAE), "Normalized RMSE": float(NRMSE), "Normalized MAE": float(NMAE)}, f)