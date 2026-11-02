import numpy as np


def influence_measures(X: np.ndarray, y: np.ndarray) -> tuple:
    """Leverage, studentized residuals and Cook's distance for every observation.

    Args:
        X (np.ndarray): (n, p) design matrix without an intercept column.
        y (np.ndarray): (n,) target.

    Returns:
        tuple: (leverage, studentized, cooks), each a length-n array.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    
    n = len(y)
    
    # 1. Construct design matrix D with intercept
    D = np.column_stack([np.ones(n), X])
    k = D.shape[1]  # k = p + 1 (number of parameters including intercept)
    
    # 2. Compute Hat Matrix diagonal elements (Leverage: h_i = H_ii)
    # H = D (D^T D)^-1 D^T
    dtd_inv = np.linalg.inv(D.T @ D)
    leverage = np.sum((D @ dtd_inv) * D, axis=1)
    
    # 3. Fit linear regression coefficients and calculate raw residuals e_i
    coef = dtd_inv @ (D.T @ y)
    y_pred = D @ coef
    residuals = y - y_pred
    
    # 4. Compute residual variance sigma^2 = RSS / (n - k)
    rss = np.sum(residuals ** 2)
    sigma_squared = rss / (n - k)
    sigma = np.sqrt(sigma_squared)
    
    # 5. Compute internally studentized residuals: r_i = e_i / (sigma * sqrt(1 - h_i))
    studentized = residuals / (sigma * np.sqrt(1.0 - leverage))
    
    # 6. Compute Cook's Distance: D_i = (r_i^2 / k) * (h_i / (1 - h_i))
    cooks = (studentized ** 2 / k) * (leverage / (1.0 - leverage))
    
    return leverage, studentized, cooks