import numpy as np

def target_neighbours(X, y, k):
    """Find the k nearest same-class points (excluding self) for each point in X.

    Args:
        X (np.ndarray): (N, D) feature matrix.
        y (np.ndarray): (N,) class labels.
        k (int): number of target neighbors.

    Returns:
        np.ndarray: (N, k) integer array of target neighbor indices.
    """
    N = X.shape[0]
    res = np.zeros((N, k), dtype=int)
    
    # Compute pairwise Euclidean distances squared
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    
    for i in range(N):
        same_class = np.where(y == y[i])[0]
        same_class = same_class[same_class != i]
        dists = dist_sq[i, same_class]
        # Sort by distance, breaking ties with smaller indices
        sorted_idx = np.lexsort((same_class, dists))
        res[i] = same_class[sorted_idx[:k]]
        
    return res

def lmnn_loss(L, X, y, k, margin, lam):
    """Compute the LMNN total loss, subgradient w.r.t. L, and number of active impostors.

    Args:
        L (np.ndarray): (r, D) linear transformation matrix.
        X (np.ndarray): (N, D) feature matrix.
        y (np.ndarray): (N,) class labels.
        k (int): number of target neighbors.
        margin (float): margin size m.
        lam (float): trade-off parameter lambda in [0, 1].

    Returns:
        tuple: (loss, grad, n_active)
    """
    N, D = X.shape
    targets = target_neighbours(X, y, k)
    
    # Transformed data
    LX = X @ L.T
    diff_LX = LX[:, np.newaxis, :] - LX[np.newaxis, :, :]
    dist_sq = np.sum(diff_LX**2, axis=-1)
    
    pull_loss = 0.0
    pull_grad = np.zeros_like(L)
    
    for i in range(N):
        for j in targets[i]:
            d2 = dist_sq[i, j]
            pull_loss += d2
            v = (X[i] - X[j]).reshape(-1, 1)
            pull_grad += 2 * (L @ (v @ v.T))
            
    push_loss = 0.0
    push_grad = np.zeros_like(L)
    n_active = 0
    
    for i in range(N):
        yi = y[i]
        impostors = np.where(y != yi)[0]
        if len(impostors) == 0:
            continue
        for j in targets[i]:
            d_ij2 = dist_sq[i, j]
            for l in impostors:
                d_il2 = dist_sq[i, l]
                hinge = margin + d_ij2 - d_il2
                if hinge > 0:
                    push_loss += hinge
                    n_active += 1
                    v_ij = (X[i] - X[j]).reshape(-1, 1)
                    v_il = (X[i] - X[l]).reshape(-1, 1)
                    push_grad += 2 * (L @ (v_ij @ v_ij.T - v_il @ v_il.T))
                    
    loss = (1.0 - lam) * pull_loss + lam * push_loss
    grad = (1.0 - lam) * pull_grad + lam * push_grad
    
    return float(loss), grad, int(n_active)