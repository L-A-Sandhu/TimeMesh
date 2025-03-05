# Timemesh Library for Loading Data with Different Normalization for Models

Timemesh is a powerful library designed to streamline the process of loading and normalizing time series data for machine learning models. With this library, you can easily load your data from CSV files, choose your desired features for input and output columns, apply different normalization techniques, and even denormalize the data for analysis. Below is a step-by-step guide explaining how you can use the Timemesh library to load and manipulate time series data.

## Table of Contents

1. [Introduction](#introduction)
2. [Loading Data with Different Normalization Techniques](#loading-data-with-different-normalization-techniques)
3. [How It Works: Time Steps and Prediction Horizon](#how-it-works-time-steps-and-prediction-horizon)
4. [Denormalizing the Data](#denormalizing-the-data)
5. [Code Example](#code-example)
6. [Verification and Summary](#verification-and-summary)

## Introduction

Timemesh simplifies loading, normalizing, and denormalizing time series data for training machine learning models. The key features include:

- **Customizable Time Steps and Prediction Horizon**: You can specify the time steps (T) and prediction horizon (H) to define how much data you want to consider in each example.
- **Multiple Normalization Techniques**: Choose from Min-Max normalization, Z-score normalization, or no normalization at all.
- **Denormalization Support**: After normalizing the data, you can easily convert it back to its original scale.
- **Flexible Input and Output Columns**: You can select any features from your data for both input and output columns, as long as their names are correctly defined.

## Loading Data with Different Normalization Techniques

Timemesh supports different normalization techniques to prepare your data for machine learning. You can choose from:

1. **Without Normalization (norm=None)**: This option leaves the data as it is, without any scaling.
2. **Min-Max Normalization (norm="MM")**: This scales the data to a fixed range, typically [0, 1].
3. **Z-Score Normalization (norm="Z")**: This normalizes the data by subtracting the mean and dividing by the standard deviation, ensuring the data has a mean of 0 and a standard deviation of 1.

## How It Works: Time Steps and Prediction Horizon

The Timemesh library uses two key parameters for defining the structure of your time series data:

- **T (Time Steps)**: This specifies how many past time steps the model will look at when making predictions.
- **H (Prediction Horizon)**: This defines how far in the future the model should predict. For instance, if you're predicting the weather for the next 6 hours, H=6.

### Example:
If you set `T=24` and `H=6`, the model will use 24 past hours of data to predict the next 6 hours.

Additionally, Timemesh allows you to define how many rows it should jump before creating a new example. This is important for ensuring the data is sampled correctly without unnecessary overlap.

## Denormalizing the Data

One of the great features of Timemesh is the ability to denormalize your data back to its original scale after normalizing it. Whether you used Min-Max normalization or Z-score normalization, you can reverse the process to get back to your raw data values.

## Code Example

Below is a complete example of how to use the Timemesh library to load, normalize, and denormalize data.

```python
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
