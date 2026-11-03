import numpy as np


def shrink_covariance(returns: np.ndarray, shrinkage: float) -> np.ndarray:
    """Linear shrinkage of the sample covariance toward a scaled identity.

    Args:
        returns (np.ndarray): (n_samples, n_assets) return matrix.
        shrinkage (float): intensity alpha in [0, 1].

    Returns:
        np.ndarray: (n_assets, n_assets) shrunk covariance matrix.
    """
    returns = np.asarray(returns)
    p = returns.shape[1]
    
    # 1. Compute sample covariance S with ddof=1
    S = np.cov(returns, rowvar=False, ddof=1)
    
    # 2. Build target F = mu * I, where mu = trace(S) / p
    mu = np.trace(S) / p
    F = mu * np.eye(p)
    
    # 3. Shrunk covariance: (1 - alpha) * S + alpha * F
    shrunk_cov = (1.0 - shrinkage) * S + shrinkage * F
    
    return shrunk_cov


def min_eigenvalue(matrix: np.ndarray) -> float:
    """Smallest eigenvalue of a symmetric matrix."""
    matrix = np.asarray(matrix)
    eigenvalues = np.linalg.eigvalsh(matrix)
    return float(eigenvalues[0])