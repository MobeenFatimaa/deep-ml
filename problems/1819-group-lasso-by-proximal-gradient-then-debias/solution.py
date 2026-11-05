import numpy as np


def group_soft_threshold(v, groups, t):
    """
    Proximal operator for the Group Lasso regularizer: 
    shrinks each group vector by max(0, 1 - t / ||vg||_2).
    """
    v = np.asarray(v, dtype=float)
    groups = np.asarray(groups, dtype=int)
    result = np.zeros_like(v)
    
    unique_groups = np.unique(groups)
    for g in unique_groups:
        idx = np.where(groups == g)[0]
        vg = v[idx]
        norm_vg = np.linalg.norm(vg)
        
        if norm_vg > t:
            scale = 1.0 - (t / norm_vg)
            result[idx] = scale * vg
        else:
            result[idx] = 0.0
            
    return result


def group_lasso_ista(X, y, groups, lam, n_iter=500, step=None):
    """
    Solves the Group Lasso problem using Proximal Gradient Descent (ISTA).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups, dtype=int)
    
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    
    if step is None:
        # Maximum singular value of X
        s_max = np.linalg.norm(X, ord=2)
        eta = 1.0 / (s_max ** 2) if s_max > 0 else 1.0
    else:
        eta = step
        
    for _ in range(n_iter):
        # Gradient of least squares loss: X^T (Xw - y)
        grad = X.T @ (X @ w - y)
        # Gradient step
        w_intermediate = w - eta * grad
        # Proximal operator step
        w = group_soft_threshold(w_intermediate, groups, eta * lam)
        
    return w


def debias(X, y, w):
    """
    Refits ordinary least squares using only the non-zero coefficients (grouped support)
    from the group lasso solution.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    
    # Identify active features (nonzero elements in w)
    active_idx = np.where(np.abs(w) > 1e-12)[0]
    
    w_debiased = np.zeros_like(w)
    if len(active_idx) > 0:
        X_active = X[:, active_idx]
        # Ordinary least squares on active columns
        w_active, _, _, _ = np.linalg.lstsq(X_active, y, rcond=None)
        w_debiased[active_idx] = w_active
        
    return w_debiased