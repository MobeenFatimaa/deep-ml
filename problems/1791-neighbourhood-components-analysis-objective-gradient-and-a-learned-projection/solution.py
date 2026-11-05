import numpy as np


def nca_objective(W, X, y):
    """
    Computes the NCA objective J(W) and its exact gradient dJ/dW.
    
    Args:
        W: projection matrix of shape (L, D)
        X: data matrix of shape (N, D)
        y: label array of shape (N,)
        
    Returns:
        (J, grad): float objective value and gradient of shape (L, D)
    """
    N, D = X.shape
    
    # Projected points
    X_proj = X @ W.T  # (N, L)
    
    # Pairwise squared distances in projected space
    diff_proj = X_proj[:, np.newaxis, :] - X_proj[np.newaxis, :, :]  # (N, N, L)
    D_sq = np.sum(diff_proj ** 2, axis=2)  # (N, N)
    np.fill_diagonal(D_sq, np.inf)
    
    # Softmax probabilities with log-sum-exp stabilization
    min_D = np.min(D_sq, axis=1, keepdims=True)
    exp_neg_D = np.exp(-(D_sq - min_D))
    np.fill_diagonal(exp_neg_D, 0.0)
    
    Z = np.sum(exp_neg_D, axis=1, keepdims=True)
    P = exp_neg_D / Z  # (N, N)
    
    # Same class mask (excluding self)
    same_class = (y[:, np.newaxis] == y[np.newaxis, :])
    np.fill_diagonal(same_class, False)
    
    # Objective J = sum_i sum_{j: y_j == y_i} p_ij
    J = float(np.sum(P[same_class]))
    
    # Gradient computation
    M1 = same_class.astype(float) * P
    P_i = np.sum(M1, axis=1, keepdims=True)  # (N, 1)
    M2 = P_i * P  # (N, N)
    np.fill_diagonal(M2, 0.0)
    
    M = M2 - M1
    Delta = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, D)
    
    # S_uv = sum_{i,j} M_ij Delta_iju Delta_ijv
    S = np.einsum('ij,iju,ijv->uv', M, Delta, Delta)
    grad = 2.0 * W @ S
    
    return J, grad


def nca_fit(X, y, W0, lr, n_iter):
    """
    Fits the linear map W using gradient ascent on the NCA objective.
    
    Args:
        X: data matrix of shape (N, D)
        y: label array of shape (N,)
        W0: initial projection matrix of shape (L, D)
        lr: learning rate
        n_iter: number of gradient ascent iterations
        
    Returns:
        (W, history): optimized projection matrix W and list of objective values (length n_iter + 1)
    """
    W = W0.copy()
    history = []
    
    J, grad = nca_objective(W, X, y)
    history.append(J)
    
    for _ in range(n_iter):
        W += lr * grad
        J, grad = nca_objective(W, X, y)
        history.append(J)
        
    return W, history