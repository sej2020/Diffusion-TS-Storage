import pickle
import pandas as pd
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import os

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
    data_df = pd.read_csv(dataset_path, encoding='unicode-escape', index_col=0)
    
    cols = data_df.columns.to_list()
    time_indices = pd.to_datetime(data_df.index, dayfirst=dayfirst)
    time_labels = [str(time) for time in time_indices]

    var_dict = {var: i for i, var in enumerate(cols)}
    time_dict = {time: i for i, time in enumerate(time_labels)}

    os.makedirs(save_folder, exist_ok=True)
    with open(f"{save_folder}/labels.pkl", 'wb') as f:
        pickle.dump((time_dict, var_dict), f)

    mean_df = data_df.mean()
    std_df = data_df.std()
    mean = mean_df.values.astype('float32')
    std = std_df.values.astype('float32')
    mean_and_std = (mean, std)

    data_df = data_df.interpolate(method='linear', limit_direction='both', axis=0)
    data = data_df.values
    data = data.astype('float32')

    with open(f'{save_folder}/data.pkl', 'wb') as f:
        pickle.dump(data, f)
    with open(f'{save_folder}/meanstd.pkl', 'wb') as f:
        pickle.dump(mean_and_std, f)



def train(model, config, train_loader, save_folder):
    
    # optimization
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_training_loss = 1e10
    for epoch in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_num, train_batch in enumerate(it, start=1):
                observed_data = train_batch['observed_data']
                presence_mask = train_batch['presence_mask']
                optimizer.zero_grad()

                loss = model(observed_data, presence_mask, is_train=1)
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
        if avg_loss < best_training_loss:
            best_training_loss = avg_loss
        else:
            print("Early stopping")
            break
    print(f"Epoch {epoch}: avg_loss {avg_loss}")

    torch.save(model.state_dict(), f"{save_folder}/model.pth")


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
                observed_data = test_batch['observed_data'] # [1, L, K]
                presence_mask = test_batch['presence_mask'] # [1, L, K]
                feature_id = test_batch['feature_id'] # [1, K]

                # [1, L, K]
                samples = model.generate(observed_data, presence_mask, feature_id, generation_variance=0)

                target_mask = 1 - presence_mask # [1, L, K]
                target = observed_data # [1, L, K]
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

        with open(f"{save_folder}/result.txt", "w") as f:
            f.write("RMSE: " + str(np.sqrt(mse_total / evalpoints_total)) + "\n")
            f.write("MAE: " + str(mae_total / evalpoints_total) + "\n")
            f.write("Normalized RMSE: " + str(np.sqrt(normalized_mse_total / evalpoints_total)) + "\n")
            f.write("Normalized MAE: " + str(normalized_mae_total / evalpoints_total) + "\n")
            print("RMSE:", np.sqrt(mse_total / evalpoints_total))
            print("MAE:", mae_total / evalpoints_total)
            print("Normalized RMSE:", np.sqrt(normalized_mse_total / evalpoints_total))
            print("Normalized MAE:", normalized_mae_total / evalpoints_total)
