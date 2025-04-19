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
parser.add_argument('--layers', type=int, default=4)
parser.add_argument('--channels', type=int, default=64)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dataset', type=str)

args = parser.parse_args()


os.makedirs(args.save_folder, exist_ok=True)
with open(f'config/{args.config}.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['data']['dataset'] = args.dataset
config['train']['device'] = args.device
config['diffusion']['layers'] = args.layers
config['diffusion']['channels'] = args.channels
config['diffusion']['nheads'] = args.channels // 8

train_loader, eval_loader, scaler, mean_scaler = get_dataloader(config=config, save_folder=args.save_folder)

model = CSDI(config, train_loader.dataset.main_data.shape[1], device=config['train']['device']).to(config['train']['device'])
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
config['total_params'] = total_params
with open(f'{args.save_folder}/config.yaml', 'w') as f:
    yaml.dump(args.__dict__, f)
    yaml.dump(config, f)

train(model, config['train'], train_loader, args.save_folder)
evaluate(model, eval_loader, scaler, args.save_folder)