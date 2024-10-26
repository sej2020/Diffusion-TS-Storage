# CSDI + Time Weaver
Fork of [CSDI](https://github.com/ermongroup/CSDI).

### Running the Experiment 

```shell
python exe_forecasting.py --nsample [number of samples for evaluation] --time_weaver
```

Note that if you want to evaluate a pre-trained model, simply specify the model folder using the `--modelfolder` argument.

### Relevant Execution Tree

*important Time Weaver modifications in italics*

`exe_forecasting.py`
- calls `get_dataloader()` from `dataset_forecasting.py`
  - *35-54 to create metadata tensor*
  - *56-78 to segment data into training, validation, and test series*
- calls `train()` from `utils.py`
  - calls `forward()` method of `CSDI_Forecasting` class in `main_model.py`
    - *300-310 to initialize new Time Weaver modules*
    - *472-488 for Time Weaver metadata processing*
  - calls `forward()` method of `diff_CSDI` class in `diff_models.py` in the `calc_loss()` method of the CSDI base class
    - *91-103 to process input data according to Time Weaver*
    - *109, 168-177 to add metadata to residual block stream according to Time Weaver*
- calls `evaluate()` from `utils.py`
  - similar to above...

### Visualize Results
'vizualize_elec_TW.ipynb' is a notebook for visualizing results.

