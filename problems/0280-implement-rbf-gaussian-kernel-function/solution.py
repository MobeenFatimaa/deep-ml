import numpy as np

def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute the RBF (Gaussian) kernel matrix between X1 and X2.
    
    Args:
        X1: First set of samples with shape (n1, d)
        X2: Second set of samples with shape (n2, d)
        gamma: Kernel coefficient (controls kernel width)
    
    Returns:
        Kernel matrix of shape (n1, n2)
    """
    X1 = np.asarray(X1, dtype=np.float64)
    X2 = np.asarray(X2, dtype=np.float64)
    
    # Compute squared Euclidean distances efficiently using matrix expansion:
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x @ y.T
    sum_sq_X1 = np.sum(X1 ** 2, axis=1, keepdims=True)  # Shape (n1, 1)
    sum_sq_X2 = np.sum(X2 ** 2, axis=1, keepdims=True).T  # Shape (1, n2)
    
    sq_dists = sum_sq_X1 + sum_sq_X2 - 2.0 * np.dot(X1, X2.T)
    
    # Ensure numerical stability by clipping negative distances due to floating-point rounding
    sq_dists = np.maximum(sq_dists, 0.0)
    
    # Apply the RBF kernel formula: K(x, y) = exp(-gamma * ||x - y||^2)
    return np.exp(-gamma * sq_dists)