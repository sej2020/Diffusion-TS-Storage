# Diffusion Time Series Storage

<img src="https://github.com/sej2020/Diffusion-TS-Storage/blob/main/wavedb2_enhanced.jpg" width="180">


Storage and analysis of time series data forms the foundation of IoT, edge computing, and personalized AI. In this paper, we present the design and architecture of a system for effectively using generative models for reducing the carbon footprint associated with time series data storage and processing. We utilize a score-based diffusion model for conditional time series generation that can replace conventional dataset storage at a fraction of the environmental impact. We intend integrate the model with a time-series database and provide low-friction interfaces for training and querying the model.

*This project is under development and can currently only support the compression of datasets that fit into RAM.*

### How to Use the Query API

You can use the `src/actions/query.py` script to query a model for regenerations of the dataset it represents. The model must be trained first using the training API. The global configuration for the query API is determined by the `config/query_config.yaml` file.

The query API can be accessed as a function (docs are in the function header) called from another script, or by the following command line utility:

```
usage: python -m src.actions.query [-h] --variables VARIABLES [VARIABLES ...] --start START --end END --freq FREQ [--n_generations N_GENERATIONS]
                [--gen_noise_magnitude GEN_NOISE_MAGNITUDE] [--n_context_features N_CONTEXT_FEATURES]

Query a model for regenerated data

options:
  -h, --help            show this help message and exit
  --variables VARIABLES [VARIABLES ...]
                        Variables to query the model for
  --start START         Start date for the query, please use format YYYY-MM-DD HH:MM:SS
  --end END             End date for the query, please use format YYYY-MM-DD HH:MM:SS
  --freq FREQ           Frequency of the data, in format <number><unit>, where unit is one of ms, s, m, h, D, W, M, Y, e.g. 1h for hourly data
  --n_generations N_GENERATIONS
                        Number of samples to generate for each data point
  --gen_noise_magnitude GEN_NOISE_MAGNITUDE
                        Spread parameter for generations. 0.0 will produce a deterministic output, 1.0 will match the model's learned distribution
  --n_context_features N_CONTEXT_FEATURES
                        Number of extra conditional features to include in the pass through the model
```

### How to Use the Training API

The training API can be used to train a model on any dataset you would like that fits into RAM. Please follow the data formatting requirements in the README file in the `data/` folder. You can find the training script at `src/actions/train.py`. To use this API, you will have to make a training configuration yaml file and put it in the `config/` folder. A couple sample training config files and a README are provided for reference.

The training API is accessible as a command line utility with the following usage:

```
usage: python -m src.actions.train [-h] --save_folder SAVE_FOLDER --config CONFIG

Training a model

options:
  -h, --help            show this help message and exit
  --save_folder SAVE_FOLDER
                        The folder name where the model and mask will be stored
  --config CONFIG       The name of the config file to use for training. Should be in the config folder.
```

### Acknowledgements

This repository builds on [CSDI](https://github.com/ermongroup/CSDI).
