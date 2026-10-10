import numpy as np

def mdn_layer(f: np.ndarray, X: np.ndarray, sigma_inv: np.ndarray, N: int) -> np.ndarray:
    """
    Apply Metadata Normalization to features.
    
    Args:
        f: Features array of shape (M, D) where M is batch size, D is feature dimension
        X: Metadata matrix of shape (M, K) where K is number of metadata variables
        sigma_inv: Pre-computed inverse covariance matrix of shape (K, K)
        N: Total number of training samples
    
    Returns:
        Residualized features of shape (M, D), orthogonal to metadata subspace
    """
    M = f.shape[0]
    cross_cov = np.dot(X.T, f)
    
    # Scale by N / M according to the MDN batch formulation
    beta = (N / M) * np.dot(sigma_inv, cross_cov)
    
    f_residual = f - np.dot(X, beta)
    
    return f_residual