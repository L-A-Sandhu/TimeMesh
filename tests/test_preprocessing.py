import pytest
import numpy as np
from timemesh import Normalizer


def test_denormalize_minmax(input_cols):
    X_norm = np.random.uniform(0, 1, size=(100, 24, len(input_cols)))
    params = {col: {"min": 0, "max": 1} for col in input_cols}

    X_denorm = Normalizer.denormalize(X_norm, params=params, method="MM", feature_order=input_cols)

    assert np.allclose(X_denorm, X_norm)


def test_denormalize_parameter_mismatch(input_cols):
    X_norm = np.random.rand(10, 24, len(input_cols))
    invalid_params = {col: {"min": 0, "max": 1} for col in input_cols}  # Now valid
    del invalid_params[input_cols[0]]["max"]  # Remove one parameter

    with pytest.raises(ValueError) as excinfo:
        Normalizer.denormalize(X_norm, params=invalid_params, method="MM", feature_order=input_cols)
    assert "Missing min/max parameters" in str(excinfo.value)


# def test_zscore_normalization(input_cols, sample_data):
#     from timemesh.preprocessing import Normalizer

#     # Test Z-score normalization
#     params = {col: {"mean": 0, "std": 1} for col in input_cols}
#     normalized = Normalizer.normalize(sample_data[input_cols].values, params, "Z", input_cols)

#     assert np.allclose(normalized.mean(axis=0), 0, atol=1e-3)
#     assert np.allclose(normalized.std(axis=0), 1, atol=1e-3)

# def test_denormalize_zscore(input_cols):
#     from timemesh.preprocessing import Normalizer

#     original = np.random.randn(100, len(input_cols))
#     params = {col: {"mean": 0, "std": 1} for col in input_cols}

#     denormalized = Normalizer.denormalize(original, params, "Z", input_cols)
#     assert np.allclose(denormalized, original)
