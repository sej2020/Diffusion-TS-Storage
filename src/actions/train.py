"""
Notes:
- we need to add data type to config that gets stored with the model
"""

import argparse
from src.utils.data import get_dataloader
import os
import yaml
from src.utils.utils import train
from src.model.main_model import CSDI

parser = argparse.ArgumentParser(description="Searching for the best parameters for compressing datasets")

def fraction_to_float(fraction: str) -> float:
    frac = fraction.split("/")
    return int(frac[0]) / int(frac[1])

# high-level options
parser.add_argument(
    "--dataset", type=str, required=True, 
    help="Please provide the dataset name (e.g. electricity, weather, etc.)"
    )
parser.add_argument(
    "--save_folder", type=str, required=True,
    help="The folder where the model and mask will be stored"
    )
parser.add_argument(
    "--device", type=str, default="cuda:0", 
    help="The device to use for training"
    )
parser.add_argument(
    "--compression", type=fraction_to_float, default=1/3,
    help="The fraction of original dataset that we keep"
    )


# hyperparameters
parser.add_argument(
    "--feature_retention_strategy", choices=["pca components", "pca loadings", "moments"], default="pca loadings", 
    help="The strategy for selecting features to retain as conditional data"
    )
parser.add_argument(
    "--history_block_size", type=int, choices=[2**i for i in range(0, 9)], default=2, 
    help="""
    The number of continuous time points in every slize of retained historical data. 
    This will not change the total number of historical data points that are preserved
    """
    )
parser.add_argument(
    "--model_param_proportion", type=float, default=0.5,
    help="""
    To compress the data, some of the info will be preserved as model weights, while some of the info will be conditional data.
    This is the proportion of the retained info that will persist as model parameters (thereby determining model size) vs. conditional data.
    E.g. 0.5 means that half of the memory dedicated to this compressed data will be model weights, while half will be real data.
    """
    )
parser.add_argument(
    "--history_to_feature_ratio", type=float, default=1.0, 
    help="The ratio of preserved points in the data coming from time slices (historical data) vs features"
    )
parser.add_argument(
    "--window_length", type=int, default=168,
    help="The time dimension of the training window"
)

# rare flags
parser.add_argument(
    "--data_dayfirst", action="store_true", # by default, this is False
    help="Whether your data csv has the day as the first element in the date string"
    )

args = parser.parse_args()

os.makedirs(args.save_folder, exist_ok=True)
train_loader, eval_loader, scaler, mean_scaler \
    = get_dataloader(
    dataset = args.dataset,
    device = args.device,
    save_folder = args.save_folder,
    compression = args.compression,
    feature_retention_strategy = args.feature_retention_strategy,
    history_block_size = args.history_block_size,
    model_param_proportion = args.model_param_proportion,
    history_to_feature_ratio = args.history_to_feature_ratio,
    window_length = args.window_length,
    data_dayfirst = args.data_dayfirst
    )

with open('config/train_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

with open(f'{args.save_folder}/config.yaml', 'w') as f:
    yaml.dump(args.__dict__, f)
    yaml.dump(config, f)

model = CSDI(config, train_loader.dataset.main_data.shape[1], args.device).to(args.device)
train(model, train_loader, eval_loader, args.save_folder)
# evaluate()