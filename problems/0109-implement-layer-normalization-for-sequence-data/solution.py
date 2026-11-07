import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Perform Layer Normalization.
    
    Parameters:
    - X: Input tensor of shape (batch_size, seq_len, features)
    - gamma: Scale parameter
    - beta: Shift parameter
    - epsilon: Small value added to variance for numerical stability
    
    Returns:
    - Normalized tensor of the same shape as X
    """
    # Compute mean and variance along the last axis (feature dimension)
    mean = np.mean(X, axis=-1, keepdims=True)
    variance = np.var(X, axis=-1, keepdims=True)
    
    # Normalize the input
    X_normalized = (X - mean) / np.sqrt(variance + epsilon)
    
    # Apply scaling (gamma) and shifting (beta)
    out = gamma * X_normalized + beta
    
    return out