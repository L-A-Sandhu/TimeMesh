import numpy as np


class DataSplitter:
    def __init__(self, X, Y, ratios):
        self.X = X
        self.Y = Y
        self.ratios = ratios
        self._validate_ratios()

    def _validate_ratios(self):
        # Check if the sum of the ratios does not exceed 100%
        total_ratio = sum(self.ratios.values())
        if total_ratio > 100:
            raise ValueError("The sum of train, test, and valid ratios cannot exceed 100%")
        elif total_ratio < 100:
            print("Warning: The sum of the ratios is less than 100%. The remaining will be unallocated.")

    def split(self):
        total_rows = len(self.X)

        # Calculate the indices for each set
        train_end_idx = int(np.floor(total_rows * self.ratios["train"] / 100))
        test_end_idx = train_end_idx + int(np.floor(total_rows * self.ratios["test"] / 100))

        # Split data
        X_train = self.X[:train_end_idx]
        Y_train = self.Y[:train_end_idx]

        X_test = self.X[train_end_idx:test_end_idx]
        Y_test = self.Y[train_end_idx:test_end_idx]

        X_valid = self.X[test_end_idx:]
        Y_valid = self.Y[test_end_idx:]

        return X_train, Y_train, X_test, Y_test, X_valid, Y_valid


# # Example usage
# X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Example data
# Y = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])  # Example labels

# # Initialize the splitter using a dictionary for the ratios
# splitter = DataSplitter(X, Y, ratios={'train': 60, 'test': 20, 'valid': 15})

# # Perform the split
# X_train, Y_train, X_test, Y_test, X_valid, Y_valid = splitter.split()

# print("Train Data:", X_train, Y_train)
# print("Test Data:", X_test, Y_test)
# print("Validation Data:", X_valid, Y_valid)
