import numpy as np


def softmax(values):
    """Computes the softmax of a 1D array or 2D matrix along the last axis with

    numerical stability.
    """
    exp_vals = np.exp(values - np.max(values, axis=-1, keepdims=True))
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


def pattern_weaver(n, crystal_values, dimension):
    """Computes the simplified self-attention pattern for a given list of crystal values.

    Args:
        n (int): Number of crystals.
        crystal_values (list or np.ndarray): Values representing the crystals.
        dimension (int): Dimensionality d_k used for scaling (d_k = dimension).

    Returns:
        np.ndarray: Weighted attention output array rounded to 4 decimal places.
    """
    # Reshape values into 2D matrix of shape (n, dimension) or (n, 1)
    X = np.array(crystal_values, dtype=float).reshape(n, -1)
    d_k = X.shape[1]

    # Compute dot-product similarity scores scaled by sqrt(d_k)
    scores = np.dot(X, X.T) / np.sqrt(d_k)

    # Compute row-wise softmax attention weights
    weights = softmax(scores)

    # Calculate final weighted contextual pattern
    output = np.dot(weights, X)

    # Flatten if 1D output is expected
    if dimension == 1:
        output = output.flatten()

    return np.round(output, 4)