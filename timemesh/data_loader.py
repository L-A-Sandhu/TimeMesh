import pandas as pd
import numpy as np
from .preprocessing import Normalizer
from .data_split import DataSplitter


class DataLoader:
    def __init__(self, ratio=None, T=1, H=1, input_cols=None, output_cols=None, step=None, norm=None):
        valid_norm_methods = [None, "MM", "Z"]
        if norm not in valid_norm_methods:
            raise ValueError(f"Invalid normalization method. Allowed: {valid_norm_methods}")

        self.T = T
        self.H = H
        self.step = step if step is not None else T
        self.norm = norm
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.normalizer = None
        self.ratio = ratio  # Corrected assignment

    def load_csv(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        except pd.errors.EmptyDataError:
            raise ValueError("CSV file is empty.")

        # Handle column defaults
        if self.input_cols is None:
            self.input_cols = list(df.columns)

        if self.output_cols is None:
            self.output_cols = [self.input_cols[-1]]  # Default output to the last column

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
        if len(df) < self.T + self.H:
            raise ValueError("Dataset is too small for the given T and H values.")

        X, Y = [], []
        max_start = len(df) - self.T - self.H + 1
        for i in range(0, max_start, self.step):
            X.append(X_data[i : i + self.T])
            Y.append(Y_data[i + self.T : i + self.T + self.H])

        X = np.array(X) if X else np.empty((0, self.T, len(self.input_cols)))
        Y = np.array(Y) if Y else np.empty((0, self.H, len(self.output_cols)))

        if X.shape[0] == 0 or Y.shape[0] == 0:
            raise ValueError("No valid (X, Y) pairs could be created. Check your T, H, and dataset size.")

        # ---- Processing Conditions ----

        # Case 1: If both `ratio` and `norm` are None → Return X, Y
        if self.norm is None and self.ratio is None:
            return X, Y

        # Case 2: If `norm` is provided but `ratio` is None → Normalize, then return X_norm, Y_norm
        if self.norm is not None and self.ratio is None:
            self.normalizer = Normalizer(method=self.norm, input_cols=self.input_cols, output_cols=self.output_cols)
            self.normalizer.fit(X, Y)
            X_norm, Y_norm = self.normalizer.transform(X, Y)
            return X_norm, Y_norm, self.normalizer.input_params, self.normalizer.output_params

        # Case 3: If `ratio` is provided but `norm` is None → Split data, then return train/test/valid sets
        if self.norm is None and isinstance(self.ratio, dict):
            splitter = DataSplitter(X, Y, ratios=self.ratio)
            return splitter.split()

        # Case 4: If both `norm` and `ratio` are provided → Normalize first, then split
        if self.norm is not None and isinstance(self.ratio, dict):
            self.normalizer = Normalizer(method=self.norm, input_cols=self.input_cols, output_cols=self.output_cols)
            self.normalizer.fit(X, Y)
            X_norm, Y_norm = self.normalizer.transform(X, Y)

            splitter = DataSplitter(X_norm, Y_norm, ratios=self.ratio)
            X_train, Y_train, X_test, Y_test, X_valid, Y_valid = splitter.split()

            return (
                X_train,
                Y_train,
                X_test,
                Y_test,
                X_valid,
                Y_valid,
                self.normalizer.input_params,
                self.normalizer.output_params,
            )
