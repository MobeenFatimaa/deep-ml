import numpy as np


def fit_poisson_irls(X: np.ndarray, y: np.ndarray, n_iter: int) -> np.ndarray:
    """Fit log(lambda) = D @ beta by iteratively reweighted least squares.

    Args:
        X (np.ndarray): (n, p) design matrix without an intercept column.
        y (np.ndarray): (n,) non-negative counts.
        n_iter (int): number of IRLS iterations, starting from beta = 0.

    Returns:
        np.ndarray: (p + 1,) coefficients, intercept first.
    """
    n, p = X.shape
    
    # Build D = [1, X] by prepending a column of ones
    ones = np.ones((n, 1), dtype=float)
    D = np.hstack([ones, X])
    
    # Initialize beta to zeros
    beta = np.zeros(p + 1, dtype=float)
    
    for _ in range(n_iter):
        eta = D @ beta
        mu = np.exp(eta)
        z = eta + (y - mu) / mu
        w = mu
        
        # Compute D^T W D efficiently without constructing an n x n matrix
        A = D.T @ (w[:, None] * D)
        
        # Compute D^T W z
        b = D.T @ (w * z)
        
        # Solve for the updated beta using np.linalg.solve
        beta = np.linalg.solve(A, b)
        
    return beta