import math
import numpy as np


def ols_inference(X: np.ndarray, y: np.ndarray) -> tuple:
    """Least-squares fit with an intercept, plus inference on each coefficient.

    Args:
        X (np.ndarray): (n, p) design matrix without an intercept column.
        y (np.ndarray): (n,) target.

    Returns:
        tuple: (coef, se, t, p), each a length-(p+1) array, intercept first.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    
    n, p = X.shape
    
    # 1. Build design matrix D by prepending a column of ones for the intercept
    D = np.column_stack([np.ones(n), X])
    
    # 2. Fit least squares coefficients: beta_hat = (D^T D)^-1 D^T y
    coef = np.linalg.lstsq(D, y, rcond=None)[0]
    
    # 3. Compute residuals and Residual Sum of Squares (RSS)
    y_pred = D @ coef
    residuals = y - y_pred
    rss = np.sum(residuals ** 2)
    
    # Degrees of freedom: n - (p + 1)
    df = n - p - 1
    
    # 4. Estimate error variance sigma^2
    sigma_squared = rss / df
    
    # 5. Compute covariance matrix of coefficients: Var(beta_hat) = sigma^2 * (D^T D)^-1
    dtd_inv = np.linalg.inv(D.T @ D)
    cov_matrix = sigma_squared * dtd_inv
    
    # Standard errors are the square roots of the diagonal elements
    se = np.sqrt(np.diagonal(cov_matrix))
    
    # 6. Compute t-statistics: t_j = beta_j / SE(beta_j)
    t = coef / se
    
    # 7. Compute two-sided p-values using vectorised math.erf:
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # p_j = 2 * (1 - Phi(|t_j|))
    erf_v = np.vectorize(math.erf)
    cdf_vals = 0.5 * (1.0 + erf_v(np.abs(t) / math.sqrt(2.0)))
    p = 2.0 * (1.0 - cdf_vals)
    
    return coef, se, t, p