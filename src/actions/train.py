import argparse
import torch
from src.utils.data import get_dataloader
import os
import warnings
import yaml
import pickle
from src.utils.utils import train, evaluate, adjust_model_architecture
from src.model.main_model import CSDI

parser = argparse.ArgumentParser(description="Training a model")

parser.add_argument(
    "--save_folder", type=str, required=True,
    help="The folder name where the model and mask will be stored"
    )

parser.add_argument(
    "--config", type=str, required=True,
    help="The name of the config file to use for training. Should be in the config folder."
)

args = parser.parse_args()


os.makedirs(args.save_folder, exist_ok=True)
with open(f'config/{args.config}.yaml', 'r') as f:
    config = yaml.safe_load(f)

if config['compression']['feature_retention_strategy'] == 'select':
    assert 'selected_features' in config['compression'], f"Please provide a selected_features list in the config file under the 'compression' section. \
        This is required for the feature retention strategy to be 'select.'"
assert config['diffusion']['channels'] / config['diffusion']['nheads'] == 8.0, "The number of channels divided by the number of heads must be 8."
assert config['compression']['data_to_model_ratio'] > 0.1, "The data to model ratio must be greater than 0.1."

# should be a db call in the future
with open(f'data/{config["data"]["dataset"]}/data.pkl', 'rb') as f:
    data = pickle.load(f)
    dataset_len, dataset_dim = data.shape
    del data

config = adjust_model_architecture(config, dataset_len, dataset_dim)

with open(f'{args.save_folder}/config.yaml', 'w') as f:
    yaml.dump(args.__dict__, f)
    yaml.dump(config, f) # dump out the config before the dataloader runs

train_loader, eval_loader, scaler, mean_scaler = get_dataloader(config=f'{args.save_folder}/config.yaml', save_folder=args.save_folder)

model = CSDI(config, train_loader.dataset.main_data.shape[1], device=config['train']['device']).to(config['train']['device'])

n_condit_points = train_loader.dataset.n_condit_points
n_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_dataset = train_loader.dataset.main_data.shape[0] * train_loader.dataset.main_data.shape[1]
warnings.warn(f"Requested compression rate: {config['compression']['compression_rate']}\nActual compression rate: {(n_condit_points + n_model_params) / total_dataset}")

# trains and saves best model
train(model, config['train'], train_loader, args.save_folder)

# load best model for evaluation
model.load_state_dict(torch.load(f"{args.save_folder}/model.pth", weights_only=True))
evaluate(model, eval_loader, scaler, args.save_folder)