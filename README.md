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



## Example Code 
Download Example dataset 
```
wget https://github.com/L-A-Sandhu/TimeMesh/blob/main/examples/data.csv
```
#### Example Usage Code 
```
import timemesh as tm
import numpy as np
import pandas as pd

# =================================================================
# Load your data for verification
# =================================================================
df = pd.read_csv("data.csv")
input_cols = ["C_WD50M", "C_WS50M", "C_PS", "C_T2M", "C_QV2M", "N_WD50M", "N_WS50M", "N_PS", "N_T2M", "N_QV2M"]
output_cols = ["C_WS50M", "N_WS50M", "S_WS50M", "E_WS50M", "W_WS50M"]

# =================================================================
# Case 1: Without Normalization (norm=None)
# =================================================================
print("\n--- Case 1: Without Normalization ---")
loader_raw = tm.DataLoader(T=24, H=6, input_cols=input_cols, output_cols=output_cols, norm=None)
X_raw, Y_raw = loader_raw.load_csv("data.csv")

print("\nLoaded raw data:")
print(f"Shape of X_raw: {X_raw.shape}")
print(f"Shape of Y_raw: {Y_raw.shape}")
print(f"First sample of X_raw:\n{X_raw[0]}")
print(f"First sample of Y_raw:\n{Y_raw[0]}")

# =================================================================
# Case 2: With Min-Max Normalization
# =================================================================
print("\n--- Case 2: With Min-Max Normalization ---")
loader_norm = tm.DataLoader(T=24, H=6, input_cols=input_cols, output_cols=output_cols, norm="MM")
X_norm, Y_norm, input_params, output_params = loader_norm.load_csv("data.csv")

print("\nLoaded normalized data:")
print(f"Shape of X_norm: {X_norm.shape}")
print(f"Shape of Y_norm: {Y_norm.shape}")
print(f"Normalization parameters (input):\n{input_params}")
print(f"Normalization parameters (output):\n{output_params}")
print(f"First sample of X_norm:\n{X_norm[0]}")
print(f"First sample of Y_norm:\n{Y_norm[0]}")

# =================================================================
# Denormalize the normalized data
# =================================================================
print("\n--- Denormalizing the normalized data ---")
X_denorm = tm.Normalizer.denormalize(
    X_norm, params=input_params, method="MM", feature_order=input_cols  # Must match original order
)

Y_denorm = tm.Normalizer.denormalize(Y_norm, params=output_params, method="MM", feature_order=output_cols)

print("\nDenormalized data:")
print(f"Shape of X_denorm: {X_denorm.shape}")
print(f"Shape of Y_denorm: {Y_denorm.shape}")
print(f"First sample of X_denorm:\n{X_denorm[0]}")
print(f"First sample of Y_denorm:\n{Y_denorm[0]}")

# =================================================================
# Verification Checks
# =================================================================
def verify_results():
    print("\n--- Verification Results ---")

    # Check 1: Raw vs Denormalized should match exactly
    x_match = np.allclose(X_raw, X_denorm, atol=1e-4)
    y_match = np.allclose(Y_raw, Y_denorm, atol=1e-4)

    print(f"X Match (Raw vs Denorm): {x_match}")
    print(f"Y Match (Raw vs Denorm): {y_match}")

    # Check 2: Normalized vs Raw ranges
    print("\nNormalization Ranges:")
    print(f"X_norm range: [{X_norm.min():.2f}, {X_norm.max():.2f}]")
    print(f"Y_norm range: [{Y_norm.min():.2f}, {Y_norm.max():.2f}]")

    # Check 3: Sample value comparison
    sample_idx = 0  # First sample
    time_idx = 0  # First timestep
    feature_idx = 1  # C_WS50M

    print("\nSample Value Comparison:")
    print(f"Original (Raw): {X_raw[sample_idx, time_idx, feature_idx]:.2f}")
    print(f"Denormalized:    {X_denorm[sample_idx, time_idx, feature_idx]:.2f}")
    print(f"Normalized:      {X_norm[sample_idx, time_idx, feature_idx]:.2f}")

verify_results()

# =================================================================
# Case 3: Test with norm=None (No normalization)
# =================================================================
def test_no_normalization():
    print("\n--- Case 3: Test with No Normalization ---")
    loader = tm.DataLoader(T=24, H=6, input_cols=input_cols, output_cols=output_cols, norm=None)
    X, Y = loader.load_csv("data.csv")

    # Directly compare with raw data from CSV
    expected_X = df[input_cols].values[:24]  # First window
    assert np.allclose(X[0], expected_X), "No normalization should return raw data"
    
    print("\nTest Passed: No normalization returns raw data successfully.")

test_no_normalization()

# =================================================================
# Case 4: With Z-Score Normalization
# =================================================================
print("\n--- Case 4: With Z-Score Normalization ---")
loader_z = tm.DataLoader(T=24, H=6, input_cols=input_cols, output_cols=output_cols, norm="Z")  # Z-score normalization
X_norm_z, Y_norm_z, input_params_z, output_params_z = loader_z.load_csv("data.csv")

print("\nLoaded Z-normalized data:")
print(f"Shape of X_norm_z: {X_norm_z.shape}")
print(f"Shape of Y_norm_z: {Y_norm_z.shape}")
print(f"Z-score Normalization parameters (input):\n{input_params_z}")
print(f"Z-score Normalization parameters (output):\n{output_params_z}")
print(f"First sample of X_norm_z:\n{X_norm_z[0]}")
print(f"First sample of Y_norm_z:\n{Y_norm_z[0]}")

# =================================================================
# Denormalize the Z-normalized data
# =================================================================
print("\n--- Denormalizing the Z-normalized data ---")
X_denorm_z = tm.Normalizer.denormalize(X_norm_z, params=input_params_z, method="Z", feature_order=input_cols)
Y_denorm_z = tm.Normalizer.denormalize(Y_norm_z, params=output_params_z, method="Z", feature_order=output_cols)

print("\nDenormalized Z-data:")
print(f"Shape of X_denorm_z: {X_denorm_z.shape}")
print(f"Shape of Y_denorm_z: {Y_denorm_z.shape}")
print(f"First sample of X_denorm_z:\n{X_denorm_z[0]}")
print(f"First sample of Y_denorm_z:\n{Y_denorm_z[0]}")

# =================================================================
# Z-Score Specific Verification
# =================================================================
def verify_zscore_results():
    print("\n--- Z-Score Specific Verification Results ---")

    # 1. Check reconstruction accuracy
    x_match = np.allclose(X_raw, X_denorm_z, atol=1e-4)
    y_match = np.allclose(Y_raw, Y_denorm_z, atol=1e-4)

    print(f"X Match (Raw vs Denorm-Z): {x_match}")
    print(f"Y Match (Raw vs Denorm-Z): {y_match}")

    # 2. Check Z-score properties
    X_flat_z = X_norm_z.reshape(-1, len(input_cols))
    print("\nZ-Score Statistics (Input Features):")
    for i, col in enumerate(input_cols):
        print(f"{col}:")
        print(f"  Mean ≈ {X_flat_z[:, i].mean():.2f} (should be ~0)")
        print(f"  Std  ≈ {X_flat_z[:, i].std():.2f} (should be ~1)")

    # 3. Sample value comparison
    sample_idx = 0
    time_idx = 0
    feature_idx = 1  # C_WS50M

    original_value = X_raw[sample_idx, time_idx, feature_idx]
    normalized_value = X_norm_z[sample_idx, time_idx, feature_idx]
    params = input_params_z[input_cols[feature_idx]]

    print("\nSample Value Breakdown (C_WS50M):")
    print(f"Original value: {original_value:.2f}")
    print(f"Normalized: ({original_value:.2f} - {params['mean']:.2f}) / {params['std']:.2f} = {normalized_value:.2f}")
    print(
        f"Denormalized: ({normalized_value:.2f} * {params['std']:.2f}) + {params['mean']:.2f} = {X_denorm_z[sample_idx, time_idx, feature_idx]:.2f}"
    )

verify_zscore_results()

# =================================================================
# Summary
# =================================================================
print("\n--- Summary ---")
print("This script has successfully run the following cases:")
print("1. Loaded raw data without normalization.")
print("2. Loaded and normalized data with Min-Max normalization.")
print("3. Denormalized the data back to the raw scale.")
print("4. Verified that the denormalized data matches the original raw data.")
print("5. Tested the case with no normalization and compared the raw data.")
print("6. Loaded and verified Z-score normalization and denormalization.")

```
