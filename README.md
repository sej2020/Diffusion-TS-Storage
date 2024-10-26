# CSDI + Time Weaver
Fork of [CSDI](https://github.com/ermongroup/CSDI), with some added elements from Time Weaver

### Running the Experiment 

```shell
python exe_forecasting.py --nsample [number of samples for evaluation] --time_weaver
```

Note that if you want to evaluate a pre-trained model, simply specify the model folder using the `--modelfolder` argument.

### Relevant Execution Tree

`exe_forecasting.py`
- calls `get_dataloader()` from `dataset_forecasting.py`
- calls `train()` from `utils.py`
  - calls `forward()` method of `CSDI_Forecasting` class in `main_model.py`
  - calls `forward()` method of `diff_CSDI` class in `diff_models.py` in the `calc_loss()` method of the CSDI base class
- calls `evaluate()` from `utils.py`
  - similar to above...

### Visualize Results
'vizualize_elec_TW.ipynb' is a notebook for visualizing results.

