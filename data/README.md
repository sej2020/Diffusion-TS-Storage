### Formatting

The training API assumes that a dataset is in a csv with the name for the dataset being the stem of the file (i.e. \<stem\>.csv). After the dataset has been processed, pkl files for this dataset will be found in the \<stem\> folder. 

We require the csv to be a TS dataset with the rows representing the time dimension and the columns representing the features. The timestamps for the series must be in the first column. If the timestamps are formatted such that the day comes first, please specify that to the training api using the `--data_dayfirst` flag.
