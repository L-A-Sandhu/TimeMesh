import pytest
import numpy as np
from timemesh.data_split import DataSplitter  # Adjust path as needed


def test_data_split():
    """Test if DataSplitter correctly splits the dataset based on given ratios."""
    # Example dataset
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    Y = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

    # Initialize DataSplitter with 60% train, 20% test, 15% validation
    ratios = {"train": 60, "test": 20, "valid": 15}
    splitter = DataSplitter(X, Y, ratios)

    # Perform the split
    X_train, Y_train, X_test, Y_test, X_valid, Y_valid = splitter.split()

    # Compute expected sizes dynamically
    total_rows = len(X)
    expected_train_size = int(np.floor(total_rows * ratios["train"] / 100))
    expected_test_size = int(np.floor(total_rows * ratios["test"] / 100))
    expected_valid_size = total_rows - expected_train_size - expected_test_size  # Adjust for remaining data

    # Validate sizes
    assert len(X_train) == expected_train_size
    assert len(Y_train) == expected_train_size
    assert len(X_test) == expected_test_size
    assert len(Y_test) == expected_test_size
    assert len(X_valid) == expected_valid_size
    assert len(Y_valid) == expected_valid_size

    # Ensure no data is lost
    combined_X = np.concatenate([X_train, X_test, X_valid])
    combined_Y = np.concatenate([Y_train, Y_test, Y_valid])

    assert np.array_equal(np.sort(combined_X), np.sort(X))
    assert np.array_equal(np.sort(combined_Y), np.sort(Y))


def test_ratio_sum_exceeds_100():
    """Ensure an error is raised when sum of ratios exceeds 100%."""
    X = np.array([1, 2, 3, 4, 5])
    Y = np.array([5, 4, 3, 2, 1])

    with pytest.raises(ValueError, match="The sum of train, test, and valid ratios cannot exceed 100%"):
        ratios = {"train": 50, "test": 30, "valid": 30}  # Sum = 110%
        DataSplitter(X, Y, ratios)


def test_warning_on_less_than_100(monkeypatch, capsys):
    """Ensure a warning is printed when the sum of ratios is less than 100%."""
    X = np.array([1, 2, 3, 4, 5])
    Y = np.array([5, 4, 3, 2, 1])

    # Set ratios to sum less than 100%
    ratios = {"train": 50, "test": 20, "valid": 10}  # Sum = 80%

    # Run DataSplitter
    splitter = DataSplitter(X, Y, ratios)
    splitter.split()

    # Capture printed output
    captured = capsys.readouterr()

    # Ensure the warning message appears
    assert "Warning: The sum of the ratios is less than 100%. The remaining will be unallocated." in captured.out
