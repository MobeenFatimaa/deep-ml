import numpy as np

def flda_fit(X, y, K):
    """Fit Multi-Class Fisher LDA and return the (D, K) projection matrix W.

    Args:
        X (np.ndarray): (n, D) data matrix.
        y (np.ndarray): (n,) integer class labels (0 to C-1).
        K (int): target projection dimensionality.

    Returns:
        np.ndarray: (D, K) projection matrix W.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    n, D = X.shape
    classes = np.unique(y)
    C = len(classes)

    if K < 1 or K > C - 1:
        raise ValueError(f"K must be between 1 and C - 1 (C={C}), got K={K}")

    # Overall mean
    mu = np.mean(X, axis=0)

    # Scatter matrices
    S_W = np.zeros((D, D), dtype=float)
    S_B = np.zeros((D, D), dtype=float)

    for c in classes:
        X_c = X[y == c]
        N_c = len(X_c)
        if N_c == 0:
            continue
        mu_c = np.mean(X_c, axis=0)
        
        # Within-class scatter
        diff = X_c - mu_c
        S_W += diff.T @ diff
        
        # Between-class scatter
        mean_diff = (mu_c - mu).reshape(-1, 1)
        S_B += N_c * (mean_diff @ mean_diff.T)

    # Symmetric inverse square root of S_W
    vals_w, vecs_w = np.linalg.eigh(S_W)
    # Ensure numerical stability for positive definite S_W
    vals_w = np.maximum(vals_w, 1e-12)
    inv_sqrt_vals_w = 1.0 / np.sqrt(vals_w)
    S_W_inv_sqrt = vecs_w @ np.diag(inv_sqrt_vals_w) @ vecs_w.T

    # Transformed matrix M = S_W^{-1/2} S_B S_W^{-1/2}
    M = S_W_inv_sqrt @ S_B @ S_W_inv_sqrt

    # Eigen-decomposition of M
    vals_m, vecs_m = np.linalg.eigh(M)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(vals_m)[::-1]
    vecs_m = vecs_m[:, sorted_indices]

    # Take the first K eigenvectors
    U = vecs_m[:, :K]

    # W = S_W^{-1/2} U
    W = S_W_inv_sqrt @ U

    # Make each column of W sign-unique: largest absolute value entry positive
    for j in range(K):
        col = W[:, j]
        max_idx = np.argmax(np.abs(col))
        if col[max_idx] < 0:
            W[:, j] = -col

    return W

def fisher_score(W, X, y):
    """Compute the Fisher criterion J(W) = |W^T S_B W| / |W^T S_W W| as a float."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    n, D = X.shape
    classes = np.unique(y)

    mu = np.mean(X, axis=0)
    S_W = np.zeros((D, D), dtype=float)
    S_B = np.zeros((D, D), dtype=float)

    for c in classes:
        X_c = X[y == c]
        N_c = len(X_c)
        if N_c == 0:
            continue
        mu_c = np.mean(X_c, axis=0)
        
        diff = X_c - mu_c
        S_W += diff.T @ diff
        
        mean_diff = (mu_c - mu).reshape(-1, 1)
        S_B += N_c * (mean_diff @ mean_diff.T)

    W = np.asarray(W, dtype=float)
    
    SB_proj = W.T @ S_B @ W
    SW_proj = W.T @ S_W @ W

    det_SB = np.linalg.det(SB_proj)
    det_SW = np.linalg.det(SW_proj)

    if abs(det_SW) < 1e-12:
        return 0.0

    return float(det_SB / det_SW)