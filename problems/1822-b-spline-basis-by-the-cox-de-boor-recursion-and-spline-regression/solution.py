import numpy as np


def bspline_basis(x, inner_knots, degree, lower, upper):
    """
    Computes the B-spline design matrix of shape (N, K + degree + 1)
    using the Cox-de Boor recursion.
    """
    x = np.atleast_1d(x)
    N = len(x)
    p = degree
    inner_knots = list(inner_knots)
    
    # Clamped knot vector t
    t = [lower] * (p + 1) + inner_knots + [upper] * (p + 1)
    t = np.array(t, dtype=float)
    
    n_basis = len(t) - p - 1
    
    def b_val(i, p_val, xi):
        if p_val == 0:
            # Last point x = upper belongs to the last non-empty interval
            if xi == t[-1]:
                return 1.0 if (t[i] <= xi <= t[i + 1]) else 0.0
            else:
                return 1.0 if (t[i] <= xi < t[i + 1]) else 0.0
        else:
            term1 = 0.0
            denom1 = t[i + p_val] - t[i]
            if denom1 > 0:
                term1 = (xi - t[i]) / denom1 * b_val(i, p_val - 1, xi)
                
            term2 = 0.0
            denom2 = t[i + p_val + 1] - t[i + 1]
            if denom2 > 0:
                term2 = (t[i + p_val + 1] - xi) / denom2 * b_val(i + 1, p_val - 1, xi)
                
            return term1 + term2

    basis = np.zeros((N, n_basis))
    for i in range(n_basis):
        for n in range(N):
            basis[n, i] = b_val(i, p, x[n])
            
    return basis


def fit_spline(x, y, n_knots, degree, lam=0.0):
    """
    Fits a B-spline regression model with optional L2 regularization (ridge penalty).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lower = float(np.min(x))
    upper = float(np.max(x))
    
    if n_knots > 0:
        q = np.linspace(0, 1, n_knots + 2)[1:-1]
        inner_knots = np.quantile(x, q).tolist()
    else:
        inner_knots = []
        
    B = bspline_basis(x, inner_knots, degree, lower, upper)
    n_basis = B.shape[1]
    
    BtB = B.T @ B
    Bty = B.T @ y
    if lam > 0.0:
        BtB += lam * np.eye(n_basis)
        
    c = np.linalg.solve(BtB, Bty)
    return c, inner_knots, lower, upper


def spline_predict(x_new, c, inner_knots, degree, lower, upper):
    """
    Predicts target values for new data points using the fitted B-spline coefficients.
    """
    x_new = np.atleast_1d(x_new)
    B = bspline_basis(x_new, inner_knots, degree, lower, upper)
    return B @ c