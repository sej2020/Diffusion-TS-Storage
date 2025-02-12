#!/bin/bash

python -m exe_forecasting --nsample 50 --history_length 0 --true_unconditional

python -m exe_forecasting --nsample 50 --history_length 1

python -m exe_forecasting --nsample 50 --history_length 2

python -m exe_forecasting --nsample 50 --history_length 3

python -m exe_forecasting --nsample 50 --history_length 6

python -m exe_forecasting --nsample 50 --history_length 12

python -m exe_forecasting --nsample 50 --history_length 24

python -m exe_forecasting --nsample 50 --history_length 48

python -m exe_forecasting --nsample 50 --history_length 72

python -m exe_forecasting --nsample 50 --history_length 96

python -m exe_forecasting --nsample 50 --history_length 120

python -m exe_forecasting --nsample 50 --history_length 144

python -m exe_forecasting --nsample 50 --history_length 168