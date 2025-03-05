import pandas as pd
import numpy as np
from .preprocessing import Normalizer


class DataLoader:
    def __init__(self, T=1, H=1, input_cols=None, output_cols=None, step=None, norm=None):
        valid_norm_methods = [None, "MM", "Z"]
        if norm not in valid_norm_methods:
            raise ValueError(f"Invalid normalization method. Allowed: {valid_norm_methods}")
        self.T = T
        self.H = H
        self.step = step if step is not None else T
        self.norm = norm
        self.input_cols = input_cols  # Store as instance variable
        self.output_cols = output_cols  # Store as instance variable
        self.normalizer = None

    def load_csv(self, csv_path):
        df = pd.read_csv(csv_path)

        # Handle column defaults (NEW CODE)
        if self.input_cols is None:
            # Exclude time columns by default
            self.input_cols = [col for col in df.columns if col not in ["YEAR", "MO", "DY", "HR"]]
        if self.output_cols is None:
            self.output_cols = self.input_cols  # Default to input columns

        # Rest of original code remains the same
        # (validation, windowing, normalization logic)

        # Validate columns
        missing_input = set(self.input_cols) - set(df.columns)
        missing_output = set(self.output_cols) - set(df.columns)
        if missing_input:
            raise ValueError(f"Missing input columns: {missing_input}")
        if missing_output:
            raise ValueError(f"Missing output columns: {missing_output}")

        # Prepare data
        X_data = df[self.input_cols].values
        Y_data = df[self.output_cols].values

        # Create windows
        X, Y = [], []
        max_start = len(df) - self.T - self.H + 1
        for i in range(0, max_start, self.step):
            X.append(X_data[i : i + self.T])
            Y.append(Y_data[i + self.T : i + self.T + self.H])

        X = np.array(X) if len(X) > 0 else np.empty((0, self.T, len(self.input_cols)))
        Y = np.array(Y) if len(Y) > 0 else np.empty((0, self.H, len(self.output_cols)))

        # Handle normalization
        if self.norm:
            self.normalizer = Normalizer(method=self.norm, input_cols=self.input_cols, output_cols=self.output_cols)
            self.normalizer.fit(X, Y)
            X_norm, Y_norm = self.normalizer.transform(X, Y)
            return (X_norm, Y_norm, self.normalizer.input_params, self.normalizer.output_params)

        return X, Y