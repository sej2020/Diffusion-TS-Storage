# python -m experiments.cut_params_expr.run_expr --save_folder 'experiments/cut_params_expr/results/electricity/layer/l4' --config 'train_config_1' --layers 4 --device 'cuda:0' --dataset 'electricity'
# python -m experiments.cut_params_expr.run_expr --save_folder 'experiments/cut_params_expr/results/electricity/layer/l3' --config 'train_config_1' --layers 3 --device 'cuda:0' --dataset 'electricity'
# python -m experiments.cut_params_expr.run_expr --save_folder 'experiments/cut_params_expr/results/electricity/layer/l2' --config 'train_config_1' --layers 2 --device 'cuda:0' --dataset 'electricity'
# python -m experiments.cut_params_expr.run_expr --save_folder 'experiments/cut_params_expr/results/electricity/layer/l1' --config 'train_config_1' --layers 1 --device 'cuda:0' --dataset 'electricity'

python -m experiments.cut_params_expr.run_expr --save_folder 'experiments/cut_params_expr/results/solar/layer/l4' --config 'train_config_1' --layers 4 --device 'cuda:0' --dataset 'solar'
python -m experiments.cut_params_expr.run_expr --save_folder 'experiments/cut_params_expr/results/solar/layer/l3' --config 'train_config_1' --layers 3 --device 'cuda:0' --dataset 'solar'
python -m experiments.cut_params_expr.run_expr --save_folder 'experiments/cut_params_expr/results/solar/layer/l2' --config 'train_config_1' --layers 2 --device 'cuda:0' --dataset 'solar'
python -m experiments.cut_params_expr.run_expr --save_folder 'experiments/cut_params_expr/results/solar/layer/l1' --config 'train_config_1' --layers 1 --device 'cuda:0' --dataset 'solar'

python -m experiments.cut_params_expr.run_expr --save_folder 'experiments/cut_params_expr/results/weather/layer/l4' --config 'train_config_1' --layers 4 --device 'cuda:0' --dataset 'weather'
python -m experiments.cut_params_expr.run_expr --save_folder 'experiments/cut_params_expr/results/weather/layer/l3' --config 'train_config_1' --layers 3 --device 'cuda:0' --dataset 'weather'
python -m experiments.cut_params_expr.run_expr --save_folder 'experiments/cut_params_expr/results/weather/layer/l2' --config 'train_config_1' --layers 2 --device 'cuda:0' --dataset 'weather'
python -m experiments.cut_params_expr.run_expr --save_folder 'experiments/cut_params_expr/results/weather/layer/l1' --config 'train_config_1' --layers 1 --device 'cuda:0' --dataset 'weather'

python -m experiments.cut_params_expr.run_expr --save_folder 'experiments/cut_params_expr/results/traffic/layer/l4' --config 'train_config_1' --layers 4 --device 'cuda:0' --dataset 'traffic'
python -m experiments.cut_params_expr.run_expr --save_folder 'experiments/cut_params_expr/results/traffic/layer/l3' --config 'train_config_1' --layers 3 --device 'cuda:0' --dataset 'traffic'
python -m experiments.cut_params_expr.run_expr --save_folder 'experiments/cut_params_expr/results/traffic/layer/l2' --config 'train_config_1' --layers 2 --device 'cuda:0' --dataset 'traffic'
python -m experiments.cut_params_expr.run_expr --save_folder 'experiments/cut_params_expr/results/traffic/layer/l1' --config 'train_config_1' --layers 1 --device 'cuda:0' --dataset 'traffic'