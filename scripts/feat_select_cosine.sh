#!/bin/bash

feature_cnt=(1 2 4 8 16 32) 

for i in {1..10}; do
    for f in ${feature_cnt[@]}; do
        python -m exe_forecasting --nsample 20 --device 'cuda:1' --n_condit_features $f --condit_strat "cosine" --out_folder "feat_select_cosine/n_feat_${f}" --train --eval
    done
done