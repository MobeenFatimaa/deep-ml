import numpy as np


def poly_fit(x, r, degree):
    """Least squares of r on 1, x, x^2, ..., x^degree.
    Returns the coefficient vector (length degree + 1, constant first).
    """
    V = np.vander(x, degree + 1, increasing=True)
    coef, _, _, _ = np.linalg.lstsq(V, r, rcond=None)
    return coef


def poly_predict(x, coef):
    """Evaluate the polynomial at x using coefficients coef."""
    V = np.vander(x, len(coef), increasing=True)
    return V @ coef


def backfit_gam(X, y, degree, n_iter=20):
    """Fit a generalized additive model using backfitting with polynomial smoothers.
    
    X is (n, p), y is (n,).
    Returns (alpha, coefs, F) where:
    - alpha: intercept (float)
    - coefs: list of p coefficient vectors
    - F: (n, p) matrix of fitted component values (centered)
    """
    n, p = X.shape
    alpha = float(np.mean(y))
    
    # Initialize component functions and their coefficients
    coefs = [np.zeros(degree + 1) for _ in range(p)]
    F = np.zeros((n, p))
    
    for _ in range(n_iter):
        for j in range(p):
            # Form partial residual: r = y - alpha - sum_{k != j} F[:, k]
            # Alternatively: r = (y - alpha) - (sum(F, axis=1) - F[:, j])
            sum_F_others = np.sum(F, axis=1) - F[:, j]
            r = y - alpha - sum_F_others
            
            # Fit polynomial to partial residual
            coef = poly_fit(X[:, j], r, degree)
            
            # Predict the component values
            f_vals = poly_predict(X[:, j], coef)
            
            # Center the fitted function values
            mean_f = np.mean(f_vals)
            f_centered = f_vals - mean_f
            
            # Adjust the constant term (coef[0]) by the mean
            coef[0] += mean_f
            
            # Update the coefficient and component matrix
            coefs[j] = coef
            F[:, j] = f_centered
            
    return alpha, coefs, F


def gam_predict(X_new, alpha, coefs):
    """Predict using alpha plus the sum of the component polynomials."""
    n_new = X_new.shape[0]
    pred = np.full(n_new, alpha)
    for j, coef in enumerate(coefs):
        pred += poly_predict(X_new[:, j], coef)
    return pred