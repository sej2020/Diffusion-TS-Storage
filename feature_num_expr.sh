#!/bin/bash

python -m exe_forecasting --nsample 50 --n_condit_features 1

python -m exe_forecasting --nsample 50 --n_condit_features 2

python -m exe_forecasting --nsample 50 --n_condit_features 4

python -m exe_forecasting --nsample 50 --n_condit_features 8

python -m exe_forecasting --nsample 50 --n_condit_features 16

python -m exe_forecasting --nsample 50 --n_condit_features 32

python -m exe_forecasting --nsample 50 --n_condit_features -1