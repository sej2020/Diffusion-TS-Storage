import pickle
import pandas as pd
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle

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

    with open(f"{save_folder}/labels.pkl", 'wb') as f:
        pickle.dump((time_dict, var_dict), f)

    mean_df = data_df.mean()
    std_df = data_df.std()
    mean = mean_df.values.astype('float32')
    std = std_df.values.astype('float32')
    mean_and_std = (mean, std)

    data_df = data_df.interpolate(method='linear', axis=0)
    data = data_df.values
    data = data.astype('float32')

    with open(f'{save_folder}/data.pkl', 'wb') as f:
        pickle.dump(data, f)
    with open(f'{save_folder}/meanstd.pkl', 'wb') as f:
        pickle.dump(mean_and_std, f)



def train(model, config, train_loader, valid_loader, save_folder, valid_epoch_interval=20):
    
    # optimization
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_num, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_num,
                        "epoch": epoch,
                    }, refresh=False)
            lr_scheduler.step()

        if (epoch + 1) % valid_epoch_interval == 0:
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_num, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_num,
                                "epoch": epoch,
                            }, refresh=False)

            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(f"\n best loss is updated to {avg_loss_valid / batch_num} at {epoch} epoch")

    torch.save(model.state_dict(), f"{save_folder}/model.pth")


def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername="", mode="test"):
    
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        normalized_mse_total = 0
        normalized_mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * ((scaler + 1e-9) ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points) 
                ) * (scaler + 1e-9)
                normalized_mse_current = mse_current / ((scaler + 1e-9) ** 2)
                normalized_mae_current = mae_current / (scaler + 1e-9)

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                normalized_mse_total += normalized_mse_current.sum().item()
                normalized_mae_total += normalized_mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )


            if mode == "test":
                with open(
                    foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
                ) as f:
                    all_target = torch.cat(all_target, dim=0)
                    all_evalpoint = torch.cat(all_evalpoint, dim=0)
                    all_observed_point = torch.cat(all_observed_point, dim=0)
                    all_observed_time = torch.cat(all_observed_time, dim=0)
                    all_generated_samples = torch.cat(all_generated_samples, dim=0)
                    pickle.dump(
                        [
                            all_generated_samples,
                            all_target,
                            all_evalpoint,
                            all_observed_point,
                            all_observed_time,
                            scaler,
                            mean_scaler,
                        ],
                        f,
                    )
                CRPS = calc_quantile_CRPS(
                    all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
                )
                CRPS_sum = calc_quantile_CRPS_sum(
                    all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
                )

                with open(
                    foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
                ) as f:
                    pickle.dump(
                        [
                            np.sqrt(mse_total / evalpoints_total),
                            mae_total / evalpoints_total,
                            CRPS,
                        ],
                        f,
                    )
                with open(foldername + "/result_nsample" + str(nsample) + ".txt", "w") as f:
                    f.write("RMSE: " + str(np.sqrt(mse_total / evalpoints_total)) + "\n")
                    f.write("MAE: " + str(mae_total / evalpoints_total) + "\n")
                    f.write("Normalized RMSE: " + str(np.sqrt(normalized_mse_total / evalpoints_total)) + "\n")
                    f.write("Normalized MAE: " + str(normalized_mae_total / evalpoints_total) + "\n")
                    f.write("CRPS: " + str(CRPS) + "\n")
                    f.write("CRPS_sum: " + str(CRPS_sum) + "\n")
                    print("RMSE:", np.sqrt(mse_total / evalpoints_total))
                    print("MAE:", mae_total / evalpoints_total)
                    print("Normalized RMSE:", np.sqrt(normalized_mse_total / evalpoints_total))
                    print("Normalized MAE:", normalized_mae_total / evalpoints_total)
                    print("CRPS:", CRPS)
                    print("CRPS_sum:", CRPS_sum)




def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):

    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):

    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1)
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)