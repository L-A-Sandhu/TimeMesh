import pandas as pd
import numpy as np

class DataLoader:
    def __init__(
        self,
        T: int = 1,
        H: int = 1,
        input_cols: list = None,
        output_cols: list = None,
        step: int = None
    ):
        self.T = T  # Input window size
        self.H = H  # Output window size
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.step = step or T  # Stride between windows

    def load_csv(self, csv_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Load CSV and create input/output windows."""
        df = pd.read_csv(csv_path)
        
        # Handle defaults
        if self.input_cols is None:
            self.input_cols = [col for col in df.columns if col not in (self.output_cols or [])]
        if self.output_cols is None:
            self.output_cols = list(df.columns)

        # Convert to numpy arrays
        X_data = df[self.input_cols].values
        Y_data = df[self.output_cols].values

        # Create windows
        X, Y = [], []
        for i in range(0, len(df) - self.T - self.H + 1, self.step):
            X_window = X_data[i:i+self.T]
            Y_window = Y_data[i+self.T:i+self.T+self.H]
            X.append(X_window)
            Y.append(Y_window.squeeze())  # Squeeze if H=1

        return np.array(X), np.array(Y)