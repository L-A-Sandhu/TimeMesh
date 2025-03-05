import pytest
import numpy as np
from timemesh.data_loader import DataLoader
import pandas as pd
from pathlib import Path


def test_default_loader(tmp_path):
    # Create sample CSV
    data = {"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]}
    df = pd.DataFrame(data)
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)

    # Load data
    loader = DataLoader(T=2, H=1)
    X, Y = loader.load_csv(csv_path)

    # Verify shapes
    assert X.shape == (2, 2, 2)  # (examples=2, T=2, input_features=2)
    assert Y.shape == (2, 2)  # (examples=2, H=1, output_features=2 → squeezed)


def test_custom_cols_and_step():
    # Test input/output column selection and step size
    pass  # You'll implement this later


def test_insufficient_rows():
    data = {"A": [1, 2], "B": [3, 4]}
    df = pd.DataFrame(data)
    csv_path = "test.csv"
    df.to_csv(csv_path, index=False)

    loader = DataLoader(T=2, H=1)
    X, Y = loader.load_csv(csv_path)
    assert len(X) == 0  # No examples possible
