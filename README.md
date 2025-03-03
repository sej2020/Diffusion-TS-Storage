# Diffusion Time Series Storage

Storage and analysis of time series data forms the foundation of IoT, edge computing, and personalized AI. In this paper, we present the design and architecture of a system for effectively using generative models for reducing the carbon footprint associated with time series data storage and processing. We utilize a score-based diffusion model for conditional time series generation that can replace conventional dataset storage at a fraction of the environmental impact. We intend integrate the model with a time-series database and provide low-friction interfaces for training and querying the model.

This project is under development and can currently only support the compression of datasets that fit into RAM.

### How to Use the Query API

You can use the `src/actions/query.py` script to query a model for regenerations of the dataset it represents. The model must be trained first using the training API. The global configuration for the query API is determined by the `config/query_config.yaml` file.

The query API can be accessed as a function (docs are in the function header) called from another script, or by the following command line utility:

```
usage: python -m src.actions.query [-h] --variables VARIABLES --start START --end END --freq FREQ [--n_generations N_GENERATIONS]
                [--generation_variance GENERATION_VARIANCE]

Query a model for regenerated data

options:
  -h, --help            show this help message and exit
  --variables VARIABLES
                        Variables to query the model for
  --start START         Start date for the query, please use format YYYY-MM-DDTHH:MM:SS, where the character T is a literal
  --end END             End date for the query, please use format YYYY-MM-DDTHH:MM:SS, where the character T is a literal
  --freq FREQ           Frequency of the data, in format <number><unit>, where unit is one of ms, s, m, h, D, W, M, Y, e.g. 1H for hourly data
  --n_generations N_GENERATIONS
                        Number of samples to generate for each data point
  --generation_variance GENERATION_VARIANCE
                        Variance of the generated data
```

### How to Use the Training API

The training API can be used to train a model on any dataset you would like that fits into RAM. Please follow the data formatting requirements in the README file in the `data/` folder. You can find the training script at `src/actions/train.py`. The global configuration for the training API is determined by the `config/train_config.yaml` file, but please do not edit any of the `diffusion` or `model` fields in the config at this moment, because flexible model configuration has not been implemented yet.

The training API is accessible as a command-line utility with the following usage:

```
usage: python -m src.actions.train [-h] --dataset DATASET --save_folder SAVE_FOLDER [--device DEVICE] [--compression COMPRESSION]
                [--feature_retention_strategy {pca components,pca loadings,moments}] [--history_block_size {1,2,4,8,16,32,64,128,256}]
                [--model_param_proportion MODEL_PARAM_PROPORTION] [--history_to_feature_ratio HISTORY_TO_FEATURE_RATIO] [--window_length WINDOW_LENGTH]
                [--data_dayfirst]

Searching for the best parameters for compressing datasets

options:
  -h, --help            show this help message and exit
  --dataset DATASET     Please provide the dataset name (e.g. electricity, weather, etc.)
  --save_folder SAVE_FOLDER
                        The folder where the model and mask will be stored
  --device DEVICE       The device to use for training
  --compression COMPRESSION
                        The fraction of original dataset that we keep
  --feature_retention_strategy {pca components,pca loadings,moments}
                        The strategy for selecting features to retain as conditional data
  --history_block_size {1,2,4,8,16,32,64,128,256}
                        The number of continuous time points in every slize of retained historical data. This will not change the total number of historical
                        data points that are preserved
  --model_param_proportion MODEL_PARAM_PROPORTION
                        To compress the data, some of the info will be preserved as model weights, while some of the info will be conditional data. This is the
                        proportion of the retained info that will persist as model parameters (thereby determining model size) vs. conditional data. E.g. 0.5
                        means that half of the memory dedicated to this compressed data will be model weights, while half will be real data.
  --history_to_feature_ratio HISTORY_TO_FEATURE_RATIO
                        The ratio of preserved points in the data coming from time slices (historical data) vs features
  --window_length WINDOW_LENGTH
                        The time dimension of the training window
  --data_dayfirst       Whether your data csv has the day as the first element in the date string
```

### Acknowledgements

This repository builds on [CSDI](https://github.com/ermongroup/CSDI).
