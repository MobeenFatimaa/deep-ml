import numpy as np

def mdn_with_collinearity(f: np.ndarray, X: np.ndarray, y: np.ndarray, 
                          sigma_tilde_inv: np.ndarray, N: int) -> np.ndarray:
    """
    Apply MDN with label collinearity control.
    
    Args:
        f: Features, shape (M, D)
        X: Metadata, shape (M, K)
        y: Labels, shape (M,) or (M, 1)
        sigma_tilde_inv: Inverse covariance of augmented [X, y] matrix, shape (K+1, K+1)
        N: Total training samples
    
    Returns:
        Features with metadata (but not label) effects removed
    """
    M = f.shape[0]
    if y.ndim == 1:
        y = y.reshape(-1, 1)
        
    # Form the augmented design matrix [X, y]
    Z = np.hstack([X, y])
    
    # Compute cross-covariance between the augmented design matrix and features
    cross_cov = np.dot(Z.T, f)
    
    # Compute joint regression coefficients using the augmented inverse covariance
    W_tilde = (N / M) * np.dot(sigma_tilde_inv, cross_cov)
    
    # Extract only the coefficients corresponding to the metadata X (the first K rows)
    K = X.shape[1]
    W_X = W_tilde[:K, :]
    
    # Remove only the metadata-related component from the features
    f_residual = f - np.dot(X, W_X)
    
    return f_residual