import argparse
import torch
import datetime
import json
import yaml
import os

from src.model.main_model import CSDI_Forecasting
from src.utils.dataset_forecasting import get_dataloader
from src.utils.utils import train, evaluate

parser = argparse.ArgumentParser(description="CSDI")
# high-level options
parser.add_argument("--train", action="store_true") # False by default
parser.add_argument("--eval", action="store_true") # False by default
# model config
parser.add_argument("--config", type=str, default="base_forecasting.yaml")
parser.add_argument("--datatype", type=str, default="electricity")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
# training options
parser.add_argument("--pseudo_unconditional", action="store_true")
parser.add_argument("--true_unconditional", action="store_true")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--time_weaver", action="store_true")
parser.add_argument("--history_length", type=int, default=168)
parser.add_argument("--pred_length", type=int, default=24)
parser.add_argument("--n_condit_features", type=int, default=-1)
# making condit strat one of a few options
parser.add_argument("--condit_strat", choices=["pca", "random", "cosine"], default="pca")
# file options
parser.add_argument("--pre_trained_model_path", type=str, default="", help="If this is null, then the model will be trained from scratch")
parser.add_argument("--out_folder", type=str, default="", help="The folder where the model will be saved")

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

assert args.train or args.eval, "Must either train or evaluate"
assert not (args.pre_trained_model_path and args.out_folder), "Cannot have both pre_trained_model_path and out_folder, out_folder is if the model is trained from scratch"
assert not (args.pseudo_unconditional and args.true_unconditional), "Cannot be both pseudo and true unconditional"
assert not ((args.pseudo_unconditional or args.true_unconditional) and args.n_condit_features > 0), "Cannot be unconditional and have conditional features"
assert args.pre_trained_model_path if not args.train else True, "If you are evaluating, you must have a pre-trained model folder"

if args.n_condit_features == 0:
    print("When n_condit_features is 0, it is the same as true_unconditional. Setting true_unconditional to True, and ignoring n_condit_features")
    args.true_unconditional = True
    args.n_condit_features = -1

config["model"]["is_pseudo_unconditional"] = args.pseudo_unconditional
config["model"]["is_true_unconditional"] = args.true_unconditional
config["model"]["history_length"] = args.history_length
config["model"]["pred_length"] = args.pred_length
config["model"]["n_condit_features"] = args.n_condit_features
config["model"]["condit_strat"] = args.condit_strat
config["model"]["condit_features"] = None
config["model"]["pred_var"] = None
config["model"]["pred_var_idx"] = None

print(json.dumps(config, indent=4))

if args.pre_trained_model_path:
    foldername = os.path.dirname(args.pre_trained_model_path)
    config = json.load(open(foldername + "/config.json"))
    if config["weaver"]["included"]:
        args.time_weaver = True
else:
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    if args.out_folder:
        foldername = "./save/" + args.out_folder + "/fc_" + args.datatype + "_" + current_time + "/"
    else:
        foldername = "./save/fc_" + args.datatype + "_" + current_time + "/"
    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)


train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    datatype=args.datatype,
    device= args.device,
    batch_size=config["train"]["batch_size"],
    eval_batch_size=config["train"]["eval_batch_size"],
    time_weaver=args.time_weaver,
    true_unconditional=args.true_unconditional,
    history_length=args.history_length,
    pred_length=args.pred_length,
    n_condit_features=args.n_condit_features,
    condit_strat=args.condit_strat,
    config=config,
)

if args.time_weaver:
    config["weaver"]["included"] = True
    config["weaver"]["k_meta"] = train_loader.dataset.metadata.shape[1]

if args.n_condit_features > 0:
    config['model']["condit_features"] = train_loader.dataset.condit_features.tolist()
    config['model']["pred_var"] = train_loader.dataset.pred_var.tolist()
    config['model']["pred_var_idx"] = train_loader.dataset.pred_var_idx.tolist()

if not args.pre_trained_model_path:
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)

target_dim = train_loader.dataset.main_data.shape[1]
model = CSDI_Forecasting(config, args.device, target_dim, time_weaver=args.time_weaver, n_condit_features=args.n_condit_features).to(args.device)

if args.train:
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        scaler=scaler,
        mean_scaler=mean_scaler,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load(args.pre_trained_model_path, weights_only=True))

model.target_dim = target_dim

if args.eval:
    evaluate(
        model,
        test_loader,
        nsample=args.nsample,
        scaler=scaler,
        mean_scaler=mean_scaler,
        foldername=foldername,
    )
