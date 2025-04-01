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
from src.utils.utils import gen_mask


def query(
    variables: list, 
    start: datetime.datetime,
    end: datetime.datetime, 
    frequency: str,
    n_generations: int = 1,
    gen_noise_magnitude: float = 1,
    n_context_features: int = 4,
    testing: bool = False
    ) -> np.ndarray:
    """
    Query a model for regenerated data. If you want to change the model or device, you will need to change the query_config.yaml file.

    Args:
        variables (List[str]): variables to query the model for
        start (datetime.datetime): start date for the query
        end (datetime.datetime): end date for the query
        frequency (str): time frequency of the data, in format <number><unit>, where unit is one of ms, s, m, h, D, W, M, Y, e.g. 1H for hourly data
        n_generations (int): number of samples to generate for each data point
        gen_noise_magnitude (float): controls the spread of the noise in the denoising step of the model. 0.0 will produce a deterministic output, 1.0 will match the spread of the model's learned distribution
        n_context_features (int): number of extra conditional features to include in the pass through the model. More features can help the model generate more accurate data,
            but can slow down the query
        testing (bool): if True, the query will also return the data slice and the presence mask slice

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
    # TEMPORARY - WILL BE GONE SOON
    with open(data_folder / 'data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    with open(model_folder / 'presence_mask_metadata.yaml', 'r') as f:
        mask_metadata = yaml.safe_load(f)
    with open(data_folder / 'labels.pkl', 'rb') as f:
        # two dictionaries: {'start': int, 'interval': int} and {str: int}
        (time_meta, var_names) = pickle.load(f)
    with open(data_folder / 'meanstd.pkl', 'rb') as f:
        (means, stds) = pickle.load(f)
    
    # nomenclature: variables == features
    query_feature_idx = [var_names[v] for v in variables]

    # getting the start and end indices
    beginning_of_data = datetime.datetime.fromtimestamp(time_meta['start'])
    start_idx = int((start - beginning_of_data).total_seconds() / time_meta['interval'])
    end_idx = int((end - beginning_of_data).total_seconds() / time_meta['interval'])

    # will have db query here in the future.
    presence_mask_slice, all_query_vars = make_query_window(
        mask_metadata = mask_metadata,
        features_in_query = query_feature_idx,
        start_idx = start_idx,
        end_idx = end_idx,
        n_context_features = n_context_features,
        model_folder = model_folder
        )
    # TEMPORARY - WILL BE IN make_query_window
    data_slice = data[start_idx:end_idx, all_query_vars]  # first n features are query features, rest are context features

    # normalizing the data
    data_slice = (data_slice - means[all_query_vars]) / (stds[all_query_vars] + 1e-9)

    # Initializing the model
    with open(model_folder / "config.yaml", "r") as f:
        model_config = yaml.safe_load(f)
    model = CSDI(model_config, dataset_dim=data.shape[1], device=DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(model_folder / "model.pth", weights_only=True))

    # model assumes batched torch tensors and will put on device for us
    data_slice = torch.tensor(data_slice).unsqueeze(0)
    presence_mask_slice = torch.tensor(presence_mask_slice).unsqueeze(0)
    all_query_vars = torch.tensor(all_query_vars).unsqueeze(0)

    # querying the model
    samples = model.generate(data_slice, presence_mask_slice, all_query_vars, n_generations, gen_noise_magnitude) 

    # denormalizing the data
    samples = samples.cpu().numpy() * (stds[all_query_vars] + 1e-9) + means[all_query_vars]

    # getting just the variables the user asked for
    query_samples = samples[:, :, :len(query_feature_idx)]

    if testing:
        denormed_data_slice = data_slice.cpu().numpy() * (stds[all_query_vars] + 1e-9) + means[all_query_vars]
        denormed_data_slice = denormed_data_slice[:, :, :len(query_feature_idx)]
        return query_samples, denormed_data_slice, presence_mask_slice.cpu().numpy()[:, :, :len(query_feature_idx)]
    else:
        return query_samples # size [n_samples, L, K]


def make_query_window(
    mask_metadata: dict,
    features_in_query: list,
    start_idx: int,
    end_idx: int,
    n_context_features: int,
    model_folder: pathlib.Path,
    ) -> tuple:
    """
    Creates a window of the mask for the presence of conditional data in the window given the query variables and the desired context size.

    Args:
        mask_metadata (dict): metadata about the presence mask
        features_in_query (list): the variables indices in the query
        start_idx (int): the start index of the window
        end_idx (int): the end index of the window
        n_context_features (int): the number of extra context features to include in the window
        model_folder (pathlib.Path): the folder where the model and presence mask metadata is saved

    Returns:
        presence_mask (np.ndarray): the presence mask for the window
        features_in_window (list): the variables indices in the window
    """
    if n_context_features > 0:
        extra_features_idx = []
        # getting top conditional features that are not in the query
        for condit_feat in mask_metadata['condit_feature_idx']:
            if condit_feat not in features_in_query:
                extra_features_idx.append(condit_feat)
            if len(extra_features_idx) == n_context_features:
                break
        features_in_window = features_in_query + extra_features_idx
    else:
        features_in_window = features_in_query
    
    # corner case: not enough conditional features to add extra context features
    if len(features_in_window) < len(features_in_query) + n_context_features:
        print(f"Warning: not enough conditional features to add extra context features. This query will only have {len(features_in_window) - len(features_in_query)} extra context features.")

    presence_mask = gen_mask(features_in_window, start_idx, end_idx, str(model_folder))

    # will not ouput features_in_window in the future, but will output data from db query
    return presence_mask, features_in_window



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Query a model for regenerated data")

    # Required
    parser.add_argument("--variables", nargs='+', type=str, required=True, help="Variables to query the model for")
    parser.add_argument("--start", type=str, required=True, help="Start date for the query, please use format YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--end", type=str, required=True, help="End date for the query, please use format YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--freq", type=str, required=True, 
        help="""
        Frequency of the data, in format <number><unit>, where unit is one of ms, s, m, h, D, W, M, Y, e.g. 1h for hourly data
        """)
    
    # Not Required
    parser.add_argument("--n_generations", type=int, default=1, help="Number of samples to generate for each data point")
    # Note: need to confirm that this is what tuning the noise parameter in model.impute actually does 
    parser.add_argument("--gen_noise_magnitude", type=float, default=1, help="Spread parameter for generations. 0.0 will produce a deterministic output, 1.0 will match the model's learned distribution")
    parser.add_argument("--n_context_features", type=int, default=4, help="Number of extra conditional features to include in the pass through the model")

    args = parser.parse_args()
    print(
        query(
        args.variables,
        datetime.datetime.fromisoformat(args.start),
        datetime.datetime.fromisoformat(args.end),
        args.freq,
        args.n_generations,
        args.gen_noise_magnitude,
        args.n_context_features
        )
    )
