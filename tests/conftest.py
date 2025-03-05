# tests/conftest.py
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_data():
    """Generate synthetic test data"""
    np.random.seed(42)
    data = {
        "C_WD50M": np.random.uniform(0, 360, 1000),
        "C_WS50M": np.random.uniform(0, 20, 1000),
        "C_PS": np.random.normal(1013, 10, 1000),
        "C_T2M": np.random.uniform(-10, 30, 1000),
        "C_QV2M": np.random.uniform(0, 0.02, 1000),
        "N_WD50M": np.random.uniform(0, 360, 1000),
        "N_WS50M": np.random.uniform(0, 20, 1000),
        "N_PS": np.random.normal(1013, 10, 1000),
        "N_T2M": np.random.uniform(-10, 30, 1000),
        "N_QV2M": np.random.uniform(0, 0.02, 1000),
        "S_WS50M": np.random.uniform(0, 20, 1000),
        "E_WS50M": np.random.uniform(0, 20, 1000),
        "W_WS50M": np.random.uniform(0, 20, 1000),
    }
    return pd.DataFrame(data)

@pytest.fixture
def input_cols():
    return [
        "C_WD50M", "C_WS50M", "C_PS", "C_T2M", "C_QV2M",
        "N_WD50M", "N_WS50M", "N_PS", "N_T2M", "N_QV2M"
    ]

@pytest.fixture
def output_cols():
    return ["C_WS50M", "N_WS50M", "S_WS50M", "E_WS50M", "W_WS50M"]
