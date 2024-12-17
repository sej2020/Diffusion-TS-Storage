#!/bin/bash

feature_cnt=(1 2 4 8 16 32) 

for f in ${feature_cnt[@]}; do
    python -m exe_forecasting --nsample 25 --device 'cuda:0' --n_condit_features $f --condit_strat 'cosine'
done