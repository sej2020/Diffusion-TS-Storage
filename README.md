# Diffusion Time Series Storage
Fork of [CSDI](https://github.com/ermongroup/CSDI)

### How to Use the Query API

You can use the `src/actions/query.py` script to query a model for regenerations of the dataset it represents. At the moment, the only usable model is in the example_model folder, but soon a training API will be added to create your own models. The gobal configuration for the query API is determined by the `config/query_config.yaml` file.

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

TBD