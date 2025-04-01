import argparse
from src.utils.data import get_dataloader
import os
import yaml
from src.utils.utils import train, evaluate
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

with open(f'{args.save_folder}/config.yaml', 'w') as f:
    yaml.dump(args.__dict__, f)
    yaml.dump(config, f)


train_loader, eval_loader, scaler, mean_scaler = get_dataloader(config_name=args.config, save_folder=args.save_folder)

model = CSDI(config, train_loader.dataset.main_data.shape[1], device=config['train']['device']).to(config['train']['device'])
train(model, config['train'], train_loader, args.save_folder)
evaluate(model, eval_loader, scaler, args.save_folder)