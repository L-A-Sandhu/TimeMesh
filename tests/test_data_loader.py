import pytest
import numpy as np
from timemesh import DataLoader


def test_dataloader_initialization():
    loader = DataLoader(T=24, H=6, input_cols=["test1", "test2"], output_cols=["target"], norm=None)
    assert loader.T == 24
    assert loader.H == 6
    assert loader.norm is None  # Changed from norm_method to norm


def test_load_csv_without_normalization(sample_data, input_cols, output_cols, tmp_path):
    loader = DataLoader(T=24, H=6, input_cols=input_cols, output_cols=output_cols, norm=None)

    # Save test data to temporary path
    test_path = tmp_path / "test_data.csv"
    sample_data.to_csv(test_path, index=False)

    X, Y = loader.load_csv(str(test_path))

    # Match the actual implementation's windowing logic
    expected_samples = len(sample_data) // loader.T  # 1000//24 = 41
    assert X.shape == (expected_samples, loader.T, len(input_cols))
    assert Y.shape == (expected_samples, loader.H, len(output_cols))


def test_invalid_normalization_method():
    with pytest.raises(ValueError) as excinfo:
        DataLoader(T=24, H=6, input_cols=["test"], output_cols=["target"], norm="invalid")
    assert "Invalid normalization method" in str(excinfo.value)


def test_mixed_normalization(sample_data, input_cols, output_cols, tmp_path):
    loader = DataLoader(T=24, H=6, input_cols=input_cols, output_cols=output_cols, norm="MM")

    test_path = tmp_path / "test_data.csv"
    sample_data.to_csv(test_path, index=False)

    X, Y, inp_params, out_params = loader.load_csv(str(test_path))
    assert "min" in inp_params[input_cols[0]]
    assert "max" in inp_params[input_cols[0]]
