import numpy as np


def fa_em_step(X, W, mu, psi):
    """Performs one EM iteration for Factor Analysis."""
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)
    mu = np.asarray(mu, dtype=float)
    psi = np.asarray(psi, dtype=float)

    N, D = X.shape
    L = W.shape[1]

    # Precompute inverse of psi (diagonal matrix)
    psi_inv = 1.0 / psi
    W_psi_inv = W * psi_inv[:, np.newaxis]  # D x L
    
    # E-step components
    # Sigma = (I_L + W^T Psi^{-1} W)^{-1}
    I_L = np.eye(L)
    Sigma = np.linalg.inv(I_L + W.T @ W_psi_inv)

    # Accumulators for M-step
    sum_xi_bi_T = np.zeros((D, L + 1))
    sum_Ci = np.zeros((L + 1, L + 1))
    sum_resid_xi_T = np.zeros(D)

    for i in range(N):
        xi = X[i]
        diff = xi - mu
        # m_i = Sigma * W^T * Psi^{-1} * (x_i - mu)
        mi = Sigma @ (W_psi_inv.T @ diff)
        
        # bi = [mi; 1]
        bi = np.empty(L + 1)
        bi[:L] = mi
        bi[L] = 1.0

        # Ci = [[Sigma + mi * mi^T, mi], [mi^T, 1]]
        mi_mi_T = np.outer(mi, mi)
        Ci = np.zeros((L + 1, L + 1))
        Ci[:L, :L] = Sigma + mi_mi_T
        Ci[:L, L] = mi
        Ci[L, :L] = mi
        Ci[L, L] = 1.0

        sum_xi_bi_T += np.outer(xi, bi)
        sum_Ci += Ci

    # M-step: W_tilde = (sum_i x_i b_i^T) * (sum_i C_i)^{-1}
    W_tilde = sum_xi_bi_T @ np.linalg.inv(sum_Ci)
    
    W_new = W_tilde[:, :L]
    mu_new = W_tilde[:, L]

    # Psi update: Psi = 1/N * diag( sum_i (x_i - W_tilde * b_i) * x_i^T )
    # Alternatively: sum_i (x_i x_i^T - W_tilde b_i x_i^T)
    # Let's compute directly using the formulation given:
    # sum_i (x_i - W_tilde * b_i) * x_i^T = sum_i (x_i x_i^T) - W_tilde * sum_i (b_i x_i^T)
    sum_xi_xi_T = X.T @ X
    sum_bi_xi_T = sum_xi_bi_T.T
    
    psi_matrix_sum = sum_xi_xi_T - W_tilde @ sum_bi_xi_T
    psi_new = np.maximum(1e-8, np.diag(psi_matrix_sum) / N)

    return W_new, mu_new, psi_new


def fa_loglik(X, W, mu, psi):
    """Computes the total log-likelihood for Factor Analysis."""
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)
    mu = np.asarray(mu, dtype=float)
    psi = np.asarray(psi, dtype=float)

    N, D = X.shape
    
    # Covariance matrix C = W W^T + Psi
    Psi = np.diag(psi)
    C = W @ W.T + Psi
    
    try:
        sign, logdet = np.linalg.slogdet(C)
        if sign <= 0:
            return -np.inf
        C_inv = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        return -np.inf

    diff = X - mu
    # Mahalanobis distances
    quad_forms = np.sum((diff @ C_inv) * diff, axis=1)
    
    const = -0.5 * D * np.log(2.0 * np.pi)
    logliks = const - 0.5 * logdet - 0.5 * quad_forms
    return float(np.sum(logliks))


def fa_em(X, L, n_iter, seed):
    """Fits Factor Analysis using the EM algorithm."""
    X = np.asarray(X, dtype=float)
    N, D = X.shape

    rng = np.random.default_rng(seed)
    W = rng.normal(size=(D, L))
    mu = np.mean(X, axis=0)
    psi = np.var(X, axis=0)
    # Prevent zero variance issues
    psi = np.maximum(1e-6, psi)

    logliks = []

    for _ in range(n_iter):
        W, mu, psi = fa_em_step(X, W, mu, psi)
        ll = fa_loglik(X, W, mu, psi)
        logliks.append(ll)

    return W, mu, psi, logliks