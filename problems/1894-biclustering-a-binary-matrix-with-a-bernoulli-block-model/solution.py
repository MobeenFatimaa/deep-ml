import numpy as np


def block_params(X, u, v, n_row, n_col, alpha=1.0, beta=1.0):
    """Computes smoothed block means theta_ab for the Bernoulli block model."""
    X = np.asarray(X, dtype=float)
    u = np.asarray(u, dtype=int)
    v = np.asarray(v, dtype=int)
    
    S = np.zeros((n_row, n_col))
    N_ab = np.zeros((n_row, n_col))
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            a = u[i]
            b = v[j]
            N_ab[a, b] += 1.0
            if X[i, j] == 1:
                S[a, b] += 1.0
                
    theta = (S + alpha) / (N_ab + alpha + beta)
    return theta


def block_loglik(X, u, v, theta):
    """Computes the total Bernoulli block model log-likelihood as a Python float."""
    X = np.asarray(X, dtype=float)
    u = np.asarray(u, dtype=int)
    v = np.asarray(v, dtype=int)
    theta = np.asarray(theta, dtype=float)
    
    eps = 1e-15
    theta_clipped = np.clip(theta, eps, 1.0 - eps)
    
    loglik = 0.0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            th = theta_clipped[u[i], v[j]]
            x = X[i, j]
            loglik += x * np.log(th) + (1.0 - x) * np.log(1.0 - th)
            
    return float(loglik)


def bicluster(X, n_row, n_col, u0, v0, n_iter):
    """Performs biclustering on a binary matrix using alternating hard assignments."""
    X = np.asarray(X, dtype=float)
    u = list(u0)
    v = list(v0)
    
    logliks = []
    eps = 1e-15
    
    for _ in range(n_iter):
        # (1) Compute theta with block_params
        theta = block_params(X, u, v, n_row, n_col)
        theta_clipped = np.clip(theta, eps, 1.0 - eps)
        
        # (2) Reassign every row i to a maximizing sum_j log p(X_ij | theta_{a, v_j})
        for i in range(len(u)):
            best_a = 0
            best_score = -np.inf
            for a in range(n_row):
                score = 0.0
                for j in range(len(v)):
                    th = theta_clipped[a, v[j]]
                    x = X[i, j]
                    score += x * np.log(th) + (1.0 - x) * np.log(1.0 - th)
                if score > best_score:
                    best_score = score
                    best_a = a
            u[i] = best_a
            
        # (3) Recompute theta
        theta = block_params(X, u, v, n_row, n_col)
        theta_clipped = np.clip(theta, eps, 1.0 - eps)
        
        # (4) Reassign every column j to b maximizing sum_i log p(X_ij | theta_{u_i, b})
        for j in range(len(v)):
            best_b = 0
            best_score = -np.inf
            for b in range(n_col):
                score = 0.0
                for i in range(len(u)):
                    th = theta_clipped[u[i], b]
                    x = X[i, j]
                    score += x * np.log(th) + (1.0 - x) * np.log(1.0 - th)
                if score > best_score:
                    best_score = score
                    best_b = b
            v[j] = best_b
            
        # (5) Recompute theta and record block_loglik
        theta = block_params(X, u, v, n_row, n_col)
        ll = block_loglik(X, u, v, theta)
        logliks.append(ll)
        
    return u, v, theta, logliks