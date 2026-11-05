import numpy as np


def lgs_posterior(mu_z, Sigma_z, W, b, Sigma_y, y):
    """
    Computes the posterior mean, posterior covariance, and log evidence
    for a linear Gaussian system.
    
    Args:
        mu_z: prior mean of shape (L,)
        Sigma_z: prior covariance of shape (L, L)
        W: observation matrix of shape (D, L)
        b: bias vector of shape (D,)
        Sigma_y: observation noise covariance of shape (D, D)
        y: observation vector of shape (D,)
        
    Returns:
        (mu_post, Sigma_post, log_evidence)
    """
    mu_z = np.asarray(mu_z, dtype=float)
    Sigma_z = np.asarray(Sigma_z, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)
    Sigma_y = np.asarray(Sigma_y, dtype=float)
    y = np.asarray(y, dtype=float)
    
    L = mu_z.shape[0]
    D = y.shape[0]
    
    # Inverses and precision matrices
    # Sigma_z_inv = np.linalg.inv(Sigma_z)
    # Sigma_y_inv = np.linalg.inv(Sigma_y)
    
    # Using linear solves for numerical stability instead of explicit matrix inversion where appropriate:
    # Sigma_z_inv = np.linalg.solve(Sigma_z, np.eye(L))
    # Sigma_y_inv = np.linalg.solve(Sigma_y, np.eye(D))
    
    # Alternatively, compute precision directly using lstsq or solve
    # Sigma_{z|y}^{-1} = Sigma_z^{-1} + W^T Sigma_y^{-1} W
    # Let's solve Sigma_y X = W => X = Sigma_y^{-1} W
    Sigma_y_inv_W = np.linalg.solve(Sigma_y, W)
    Sigma_z_inv = np.linalg.solve(Sigma_z, np.eye(L))
    
    Sigma_post_inv = Sigma_z_inv + W.T @ Sigma_y_inv_W
    Sigma_post = np.linalg.solve(Sigma_post_inv, np.eye(L))
    
    # mu_{z|y} = Sigma_{z|y} [ W^T Sigma_y^{-1} (y - b) + Sigma_z^{-1} mu_z ]
    diff = y - b
    term1 = W.T @ np.linalg.solve(Sigma_y, diff)
    term2 = Sigma_z_inv @ mu_z
    mu_post = Sigma_post @ (term1 + term2)
    
    # Evidence: p(y) = N(y | W mu_z + b, Sigma_y + W Sigma_z W^T)
    mean_y = W @ mu_z + b
    cov_y = Sigma_y + W @ Sigma_z @ W.T
    
    # Log probability of y under N(mean_y, cov_y)
    # log N(y | m, S) = -0.5 * [ D * log(2pi) + log|S| + (y - m)^T S^{-1} (y - m) ]
    sign, log_det = np.linalg.slogdet(cov_y)
    inv_cov_y_diff = np.linalg.solve(cov_y, y - mean_y)
    quadratic_form = (y - mean_y).T @ inv_cov_y_diff
    
    log_evidence = -0.5 * (D * np.log(2.0 * np.pi) + log_det + quadratic_form)
    
    return mu_post, Sigma_post, float(log_evidence)