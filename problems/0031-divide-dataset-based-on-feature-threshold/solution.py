import numpy as np


def divide_on_feature(X: np.ndarray, feature_i: int, threshold: float | int):
    """Divide a dataset into two subsets based on whether a specified feature's value

    is greater than or equal to a given threshold.
    """
    # Create a boolean mask for samples where the specified feature >= threshold
    mask = X[:, feature_i] >= threshold

    # Filter the array into two subsets
    X_ge = X[mask]
    X_lt = X[~mask]

    return [X_ge, X_lt]