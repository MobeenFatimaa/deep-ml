import numpy as np

def instance_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Perform Instance Normalization over a 4D tensor X of shape (B, C, H, W).
    gamma: scale parameter of shape (C,)
    beta: shift parameter of shape (C,)
    epsilon: small value for numerical stability
    Returns: normalized array of same shape as X
    """
    B, C, H, W = X.shape
    
    # 1. Compute mean and variance independently for each instance and channel across (H, W)
    mean = np.mean(X, axis=(2, 3), keepdims=True)
    var = np.var(X, axis=(2, 3), keepdims=True)
    
    # 2. Normalize the spatial dimensions
    X_norm = (X - mean) / np.sqrt(var + epsilon)
    
    # 3. Apply learned scale (gamma) and shift (beta), reshaped for broadcasting
    gamma_reshaped = gamma.reshape(1, C, 1, 1)
    beta_reshaped = beta.reshape(1, C, 1, 1)
    
    return X_norm * gamma_reshaped + beta_reshaped