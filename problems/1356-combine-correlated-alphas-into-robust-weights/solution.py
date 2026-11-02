import numpy as np


def combine_alphas(mu: np.ndarray, cov: np.ndarray, shrinkage: float) -> np.ndarray:
    """Mean-variance weights from shrunk covariance, normalised to sum to 1.

    Args:
        mu (np.ndarray): (p,) expected returns of each alpha.
        cov (np.ndarray): (p, p) covariance matrix of the alphas.
        shrinkage (float): intensity alpha in [0, 1].

    Returns:
        np.ndarray: (p,) weights summing to 1.
    """
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    p = len(mu)
    
    # 1. Compute mean variance / average variance as the trace divided by p
    mean_var = np.trace(cov) / p
    
    # 2. Shrink covariance matrix toward a scaled identity:
    # \hat{\Sigma} = (1 - \alpha) * cov + \alpha * \bar{\mu} * I
    # (Note: Using mean_var as the scalar multiplier for the identity matrix)
    shrunk_cov = (1.0 - shrinkage) * cov + shrinkage * mean_var * np.eye(p)
    
    # 3. Solve for unnormalised weights: w_raw = \hat{\Sigma}^{-1} mu
    # Using np.linalg.solve is numerically safer than explicitly inverting the matrix.
    w_raw = np.linalg.solve(shrunk_cov, mu)
    
    # 4. Normalise weights to sum to 1
    w = w_raw / np.sum(w_raw)
    
    return w