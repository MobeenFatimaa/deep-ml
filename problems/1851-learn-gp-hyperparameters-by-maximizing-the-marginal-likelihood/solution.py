from math import log, pi
import numpy as np


def neg_log_marglik_and_grad(theta, X, y):
    """
    Computes the negative log marginal likelihood (NLL) and its gradient 
    with respect to log hyperparameters theta = (log ell, log sigma_f, log sigma_y).
    """
    if X.ndim == 1:
        X = X[:, np.newaxis]
    N = X.shape[0]
    
    ell = np.exp(theta[0])
    sigma_f = np.exp(theta[1])
    sigma_y = np.exp(theta[2])
    
    # Pairwise squared distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    
    K = (sigma_f ** 2) * np.exp(-dist_sq / (2.0 * (ell ** 2)))
    K_sigma = K + (sigma_y ** 2) * np.eye(N)
    
    # Cholesky decomposition for numerical stability
    try:
        L = np.linalg.cholesky(K_sigma)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(K_sigma + 1e-6 * np.eye(N))
        
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    
    # NLL = 0.5 * y^T alpha + log |L| + 0.5 * N * log(2*pi)
    nll = 0.5 * np.dot(y, alpha) + np.sum(np.log(np.diag(L))) + 0.5 * N * log(2.0 * pi)
    
    # Inverse of K_sigma
    K_inv = np.linalg.inv(L.T) @ np.linalg.inv(L)
    
    # A = alpha alpha^T - K_sigma^{-1}
    A = np.outer(alpha, alpha) - K_inv
    
    # Derivatives of K_sigma w.r.t log parameters
    dK_dtheta0 = K * (dist_sq / (ell ** 2))
    dK_dtheta1 = 2.0 * K
    dK_dtheta2 = 2.0 * (sigma_y ** 2) * np.eye(N)
    
    # Gradient of NLL = - Gradient of log p
    grad0 = -0.5 * np.sum(A * dK_dtheta0)
    grad1 = -0.5 * np.sum(A * dK_dtheta1)
    grad2 = -0.5 * np.sum(A * dK_dtheta2)
    
    return float(nll), np.array([grad0, grad1, grad2])


def fit_hyperparameters(X, y, inits, n_steps=200, lr=0.05):
    """
    Fits GP hyperparameters using gradient descent on the negative log marginal likelihood
    starting from multiple initialization points.
    """
    best_theta = None
    best_nll = float('inf')
    final_nlls = []
    
    for init in inits:
        theta = np.array(init, dtype=float)
        for _ in range(n_steps):
            _, grad = neg_log_marglik_and_grad(theta, X, y)
            theta -= lr * grad
            
        final_nll, _ = neg_log_marglik_and_grad(theta, X, y)
        final_nlls.append(final_nll)
        
        if final_nll < best_nll:
            best_nll = final_nll
            best_theta = theta
            
    return best_theta, best_nll, final_nlls