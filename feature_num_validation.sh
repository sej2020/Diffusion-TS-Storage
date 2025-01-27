#!/bin/bash

# feature_cnt=(1 2 4 8 16 32) 

# for f in ${feature_cnt[@]}; do
#     python -m exe_forecasting --nsample 25 --device 'cuda:1' --n_condit_features 64 
# done

for i in {1..3}; do
    # python -m exe_forecasting --nsample 25 --device 'cuda:1' --n_condit_features 64 --condit_strat 'random' --out_folder 'feat_num_val' --train --eval
    python -m exe_forecasting --nsample 25 --device 'cuda:1' --out_folder 'feat_num_val' --train --eval
    python -m exe_forecasting --nsample 25 --device 'cuda:1' --out_folder 'feat_num_val' --train --eval --true_unconditional
done