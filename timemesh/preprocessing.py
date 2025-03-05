import numpy as np
from typing import Dict, List, Literal, Tuple


class Normalizer:
    def __init__(self, method: Literal["MM", "Z"], input_cols: List[str], output_cols: List[str]):
        self.method = method
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.input_params: Dict[str, Dict[str, float]] = {}
        self.output_params: Dict[str, Dict[str, float]] = {}

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Learn normalization parameters"""
        X_flat = X.reshape(-1, len(self.input_cols))
        for i, col in enumerate(self.input_cols):
            if self.method == "MM":
                self.input_params[col] = {"min": X_flat[:, i].min(), "max": X_flat[:, i].max()}
            else:
                self.input_params[col] = {"mean": X_flat[:, i].mean(), "std": X_flat[:, i].std()}

        Y_flat = Y.reshape(-1, len(self.output_cols)) if Y.ndim == 3 else Y
        for i, col in enumerate(self.output_cols):
            if self.method == "MM":
                self.output_params[col] = {"min": Y_flat[:, i].min(), "max": Y_flat[:, i].max()}
            else:
                self.output_params[col] = {"mean": Y_flat[:, i].mean(), "std": Y_flat[:, i].std()}

    def transform(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply normalization"""
        X_norm = X.copy()
        Y_norm = Y.copy()

        X_flat = X_norm.reshape(-1, len(self.input_cols))
        for i, col in enumerate(self.input_cols):
            params = self.input_params[col]
            if self.method == "MM":
                X_flat[:, i] = (X_flat[:, i] - params["min"]) / (params["max"] - params["min"] + 1e-8)
            else:
                X_flat[:, i] = (X_flat[:, i] - params["mean"]) / (params["std"] + 1e-8)

        Y_flat = Y_norm.reshape(-1, len(self.output_cols)) if Y_norm.ndim == 3 else Y_norm
        for i, col in enumerate(self.output_cols):
            params = self.output_params[col]
            if self.method == "MM":
                Y_flat[:, i] = (Y_flat[:, i] - params["min"]) / (params["max"] - params["min"] + 1e-8)
            else:
                Y_flat[:, i] = (Y_flat[:, i] - params["mean"]) / (params["std"] + 1e-8)

        return X_norm, Y_norm

    @staticmethod
    def denormalize(
        data: np.ndarray, params: Dict[str, Dict[str, float]], method: Literal["MM", "Z"], feature_order: List[str]
    ) -> np.ndarray:
        """Denormalize with explicit feature order"""
        for col in feature_order:
            if col not in params:
                raise KeyError(f"Missing parameters for {col}")

            if method == "MM" and ("min" not in params[col] or "max" not in params[col]):
                raise ValueError(f"Missing min/max parameters for {col}")
            if method == "Z" and ("mean" not in params[col] or "std" not in params[col]):
                raise ValueError(f"Missing mean/std parameters for {col}")

        data = data.copy()
        original_shape = data.shape
        data_flat = data.reshape(-1, len(feature_order))

        for i, col in enumerate(feature_order):
            col_params = params[col]
            if method == "MM":
                data_flat[:, i] = (data_flat[:, i] * (col_params["max"] - col_params["min"])) + col_params["min"]
            else:
                data_flat[:, i] = (data_flat[:, i] * col_params["std"]) + col_params["mean"]

        return data_flat.reshape(original_shape)
