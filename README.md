# Timemesh 🕰️

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![PyPI Version](https://img.shields.io/pypi/v/timemesh)](https://pypi.org/project/timemesh/)

A Python library for efficient time series data preprocessing and windowing for machine learning.

## Features

- 🚀 **Flexible Windowing**: Create overlapping/non-overlapping windows with configurable time steps (T) and horizon (H)
- 📊 **Normalization**: Supports Min-Max and Z-score normalization
- 🔄 **Denormalization**: Revert normalized data back to original scale
- 🧩 **Modular Design**: Separate data loading and normalization logic
- ✅ **Validation**: Built-in data integrity checks

## Installation

```bash
pip install timemesh
```

## Quick Start
```
import timemesh as tm

# Initialize data loader
loader = tm.DataLoader(
    T=24,  # Use 24 historical steps
    H=6,   # Predict 6 steps ahead
    input_cols=["temperature", "humidity"],
    output_cols=["target_feature"],
    norm="MM"  # Min-Max normalization
)

# Load and preprocess data
X, Y, input_params, output_params = loader.load_csv("data.csv")
```
| Parameter     | Description                          | Default | Options               |
|---------------|--------------------------------------|---------|-----------------------|
| **T**         | Historical time steps per sample     | 1       | Any positive integer  |
| **H**         | Prediction horizon steps             | 1       | Any positive integer  |
| **input_cols**| Features used for model input        | None(All will be input)       | List of column names  |
| **output_cols**| Target features for prediction      | None(All Will be output        | List of column names  |
| **norm**      | Normalization method                 | `None`(No Normalization)  | `"MM"`, `"Z"`         |
| **steps**     | Step size between windows            | `None`(Non overlapping)  | Any positive integer  |



## Normalization Workflow
#### Load with Min-Max normalization
loader = tm.DataLoader(..., norm="MM")
X_norm, Y_norm, in_params, out_params = loader.load_csv("data.csv")

# Denormalize predictions
```
denormalized = tm.Normalizer.denormalize(
    Y_norm, 
    params=out_params,
    method="MM",
    feature_order=["target"]
)
```
##
