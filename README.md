# Diffusion DB
Fork of [CSDI](https://github.com/ermongroup/CSDI)

### Running the Experiment 

TBD

### Relevant Execution Tree

TBD

`exe_forecasting.py`
- calls `get_dataloader()` from `dataset_forecasting.py`
- calls `train()` from `utils.py`
  - calls `forward()` method of `CSDI_Forecasting` class in `main_model.py`
  - calls `forward()` method of `diff_CSDI` class in `diff_models.py` in the `calc_loss()` method of the CSDI base class
- calls `evaluate()` from `utils.py`
  - similar to above...

### Visualize Results

TBD

'vizualize_elec_TW.ipynb' is a notebook for visualizing results.

