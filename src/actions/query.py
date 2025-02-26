"""
End user can query a model for regenerated data. Can be run from the command line or called from another script.
"""
import argparse
import datetime
import yaml
import pathlib
from src.model.main_model import CSDI
import torch
import pickle
import numpy as np


def query(
    variables: list, 
    start: datetime.datetime,
    end: datetime.datetime, 
    frequency: str,
    n_generations: int = 1,
    generation_variance: float = 1
    ) -> np.ndarray:
    """
    Query a model for regenerated data. If you want to change the model or device, you will need to change the query_config.yaml file.

    Args:
        variables (List[str]): variables to query the model for
        start (datetime.datetime): start date for the query
        end (datetime.datetime): end date for the query
        frequency (str): time frequency of the data, in format <number><unit>, where unit is one of ms, s, m, h, D, W, M, Y, e.g. 1H for hourly data
        n_generations (int): number of samples to generate for each data point
        generation_variance (float): variance of the generated data

    Returns:
        out (np.ndarray): the regenerated data, with shape [_num_samples_, _time_, _num_variables_]
    """
    print(f"Querying model for variable(s) {variables} between {start} and {end} with frequency {frequency}")
    print(f"Note: ignoring frequency for now")

    # Opening global config and getting everything we need
    with open("config/query_config.yaml", "r") as f:
        global_config = yaml.safe_load(f)
    model_folder = pathlib.Path(global_config['model_folder'])
    data_folder = pathlib.Path(global_config['data_folder'])
    DEVICE = global_config["device"] if torch.cuda.is_available() else "cpu"
    
    # Getting the conditional data for the model
    # Note: as of right now, this will be full data and the prediction mask
    with open(data_folder / 'data.pkl', 'rb') as f:
        # full data now, but in the future this will load just conditional data
        data = pickle.load(f)
    with open(model_folder / 'presence_mask.pkl', 'rb') as f:
        # mask to make full data just conditional data. unclear if this will be necessary in the future
        presence_mask = pickle.load(f)
    with open(data_folder / 'labels.pkl', 'rb') as f:
        # two dictionaries: {'YYYY-MM-DD HH:MM:SS': int} and {str: int}
        (time_stamps, var_names) = pickle.load(f)
    with open(data_folder / 'meanstd.pkl', 'rb') as f:
        (means, stds) = pickle.load(f)
    
    # Getting the window: for now, the window size is only that of the query
    # in the future, we will expand the size of the window based on experiments about how much conditional data is most useful
    var_slice = [var_names[v] for v in variables]
    start_idx = time_stamps[start]
    end_idx = time_stamps[end]
    data_slice = data[start_idx:end_idx, var_slice]
    presence_mask_slice = presence_mask[start_idx:end_idx, var_slice]

    # normalizing the data
    data_slice = (data_slice - means[var_slice]) / (stds[var_slice] + 1e-9)

    # Initializing the model
    with open(model_folder / "config.json", "r") as f:
        model_config = yaml.safe_load(f)
    model = CSDI(model_config, dataset_dim=data.shape[1], device=DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(model_folder / "model.pth", weights_only=True))

    # model assumes batched torch tensors and will put on device for us
    data_slice = torch.tensor(data_slice).unsqueeze(0)
    presence_mask_slice = torch.tensor(presence_mask_slice).unsqueeze(0)
    var_slice = torch.tensor(var_slice).unsqueeze(0)

    # querying the model
    samples = model.generate(data_slice, presence_mask_slice, var_slice, n_generations, generation_variance) 

    # denormalizing the data
    samples = samples.cpu().numpy() * (stds[var_slice] + 1e-9) + means[var_slice]

    # of size [num_samples, L, K]
    return samples



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Query a model for regenerated data")
    def list_of_vars(string):
        return [var.strip() for var in string.split(",")]

    # Required
    parser.add_argument("--variables", type=list_of_vars, required=True, help="Variables to query the model for")
    parser.add_argument("--start", type=str, required=True, help="Start date for the query, please use format YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--end", type=str, required=True, help="End date for the query, please use format YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--freq", type=str, required=True, 
        help="""
        Frequency of the data, in format <number><unit>, where unit is one of ms, s, m, h, D, W, M, Y, e.g. 1H for hourly data
        """)
    
    # Not Required
    parser.add_argument("--n_generations", type=int, default=1, help="Number of samples to generate for each data point")
    # Note: need to confirm that this is what tuning the noise parameter in model.impute actually does 
    parser.add_argument("--generation_variance", type=float, default=1, help="Variance of the generated data")

    args = parser.parse_args()
    print(
        query(
        args.variables,
        args.start,
        args.end,
        args.freq,
        args.n_generations,
        args.generation_variance
        )
    )
