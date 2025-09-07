# [Pattern Recognition] Decomposition Dynamic Graph Conolutional Recurrent Network for Traffic Forecasting  

This is a PyTorch implementation of D3MG-3DP: A Decomposition-Aware Dynamic Multi-Graph Network with 3D Convolutional Priors for Traffic Forecasting

# Data Preparation

For convenience, we package these datasets used in our model in [Google Drive](https://drive.google.com/drive/folders/1OQYVddI5icsHwSVWtRHbqJ-xG7242q1r?usp=share_link).

Unzip the downloaded dataset files to the main file directory, the same directory as run.py.

# Requirements

Python 3.6.5, Pytorch 1.9.0, Numpy 1.16.3, argparse and configparser

# Model Training

```bash
python run.py --dataset {DATASET_NAME} --mode {MODE_NAME} --model_name {MODEL_NAME} 
```
Replace `{DATASET_NAME}` with one of `PEMSD3`, `PEMSD4`, `PEMSD7`, `PEMSD8`, `PEMSD7(L)`, `PEMSD7(M)`

such as `python run.py --dataset PEMSD4`

There are two options for `{MODEL_NAME}` : `DDGCRN` and `D3MG_3DP`

There are two options for `{MODE_NAME}` : `train` and `test`

Selecting `train` will retrain the model and save the trained model parameters and records in the `experiment` folder.

For `test` you must put your model absolute path in load_path variable in run.py