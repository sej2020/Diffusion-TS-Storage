
# Training Config File Format
```
data:
  dataset (str): # the dataset name (e.g. electricity, weather, etc.)
  day_first (bool): # whether your data csv has the day as the first element in the date string

compression:
  compression_rate (float): # the fraction of original dataset that we keep
  feature_retention_strategy (str): # the strategy for selecting features to retain as conditional data. OPTIONS = {'pca', 'select'}
  history_block_size (int): # the number of continuous time points in every slice of retained historical data; this will not change the total number of historical data points that are preserved
  data_to_model_ratio (float): # To compress the data, some of the info will be preserved as model weights, while some of the info will be conditional data. This is the proportion of the retained info that will persist as conditional data vs. model parameters (thereby determining model size). E.g. 1.0 means that half of the memory dedicated to this compressed dataset will be real data, while half will be model weights.
  history_to_feature_ratio (float): # the ratio of preserved points in the data coming from time slices (historical data) vs. features
  selected_features (list[str]): # If your feature_retention_strategy is 'select', you will have to provide this list of the names of the features you would like to preserve. If there is remaining capacity under the given compression rate, remaining features will be selected by the model.

train:
  epochs (int)
  batch_size (int): # 8 or 16 is recommended
  lr (float): # 1.0e-3 is recommended
  window_length (int): # the time dimension of the training window
  device (str): # cpu, cuda, or cuda: #0, etc.

diffusion:
  layers (int): # diffusion model residual layers - 4 is recommended 
  channels (int): # the internal dimension of the model - 64 is recommended
  nheads (int): # number of attention heads in the transformer layers - 8 is recommended
  diffusion_embedding_dim (int): # 128 is strongly recommended
  beta_start (float): # noise schedule starting value - 0.0001 is strongly recommended
  beta_end (float): # noise schedule ending value - 0.5 is strongly recommended
  num_steps (int): # number of diffusion noising/denoising steps - 50 is recommended
  schedule (str): # linear or quadratic spacing of the noise schedule - OPTIONS = {"quad", "linear"} - quad is recommended
  is_linear (bool): # whether to use linear transformer layers - True is recommended

model:
  timeemb (int): # time embedding dimension - 128 is recommended
  featureemb (int): # feature embedding dimension - 16 is recommended
  training_feature_sample_size (int): # number of features sampled for training at each step - 64 is recommended
```