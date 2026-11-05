from math import log, pi
import numpy as np


def ard_fit(X, y, sigma2, n_iter, alpha_cap=1e8):
    """
    Automatic Relevance Determination (ARD) for linear regression using MacKay's fixed-point updates.

    Args:
        X: design matrix of shape (N, D)
        y: target vector of shape (N,)
        sigma2: known noise variance (float)
        n_iter: number of fixed-point iterations
        alpha_cap: maximum precision cap for pruning (default 1e8)

    Returns:
        (m, alpha, log_ev): 
            - m: final posterior mean of shape (D,) recomputed with final alpha
            - alpha: final precision vector of shape (D,)
            - log_ev: list of log evidences log p(y|X, alpha) at the start of each iteration
    """
    N, D = X.shape
    alpha = np.ones(D, dtype=float)
    log_ev = []

    for _ in range(n_iter):
        A = np.diag(alpha)
        # Compute posterior covariance Sigma and mean m
        # Sigma = (A + X^T X / sigma2)^-1
        Sigma = np.linalg.inv(A + (X.T @ X) / sigma2)
        m = (Sigma @ X.T @ y) / sigma2

        # Evaluate log evidence log p(y|X, alpha) under current alpha
        # C = sigma2 * I_N + X A^-1 X^T
        # log|C| = N*log(sigma2) - log|Sigma| - sum(log(alpha))
        _, logdet_Sigma = np.linalg.slogdet(Sigma)
        logdet_C = N * np.log(sigma2) - logdet_Sigma - np.sum(np.log(alpha))
        
        # y^T C^-1 y = y^T (y - X m) / sigma2
        quad = np.dot(y, y - X @ m) / sigma2
        log_evidence = -0.5 * (N * np.log(2.0 * pi) + logdet_C + quad)
        log_ev.append(float(log_evidence))

        # MacKay fixed-point update:
        # gamma_j = 1 - alpha_j * Sigma_jj
        # alpha_j <- gamma_j / m_j^2
        gamma = 1.0 - alpha * np.diag(Sigma)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            new_alpha = gamma / (m ** 2)
            new_alpha = np.where((m ** 2 > 1e-20) & (gamma > 0), new_alpha, alpha_cap)

        # Clip alpha_j to alpha_cap
        alpha = np.clip(new_alpha, 0.0, alpha_cap)

    # Recompute posterior mean m using the final alpha
    A = np.diag(alpha)
    Sigma = np.linalg.inv(A + (X.T @ X) / sigma2)
    m = (Sigma @ X.T @ y) / sigma2

    return m, alpha, log_ev