import numpy as np

def _multivariate_normal_logpdf(X, mean, cov, reg_covar=1e-6):
    """
    Compute the log pdf of a multivariate normal distribution with numerical stability.
    X: (N, D)
    mean: (D,)
    cov: (D, D)
    """
    N, D = X.shape
    # Ensure covariance is symmetric and add regularization if needed
    cov = 0.5 * (cov + cov.T)
    
    try:
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            # Fallback if non-positive definite
            cov = cov + np.eye(D) * reg_covar
            sign, logdet = np.linalg.slogdet(cov)
            
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov = cov + np.eye(D) * reg_covar
        sign, logdet = np.linalg.slogdet(cov)
        inv_cov = np.linalg.inv(cov + np.eye(D) * reg_covar)
        
    diff = X - mean  # (N, D)
    # Quadratic term: sum((diff @ inv_cov) * diff, axis=1)
    quad = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
    
    constant = -0.5 * (D * np.log(2 * np.pi) + logdet)
    return constant - 0.5 * quad

def gmm_em(X, means0, cov_type, n_iter):
    """
    Fit a Gaussian mixture by EM with one of four covariance structures.
    """
    N, D = X.shape
    K = means0.shape[0]
    
    # 1. Initialize weights pi, means, and global data covariance S
    pi = np.ones(K) / K
    means = np.array(means0, dtype=float)
    
    # Initial S from data (divisor N)
    diff_data = X - np.mean(X, axis=0)
    S_global = np.dot(diff_data.T, diff_data) / N
    
    # Initialize covariances based on cov_type
    covs = np.zeros((K, D, D))
    for k in range(K):
        if cov_type == 'full':
            covs[k] = np.copy(S_global)
        elif cov_type == 'tied':
            covs[k] = np.copy(S_global)
        elif cov_type == 'diag':
            covs[k] = np.diag(np.diag(S_global))
        elif cov_type == 'spherical':
            tr_S = np.trace(S_global) / D
            covs[k] = tr_S * np.eye(D)
        
        # Add 10^-6 to diagonal
        covs[k] += np.eye(D) * 1e-6

    reg_covar = 1e-6
    
    for _ in range(n_iter):
        # --- E Step ---
        log_prob = np.zeros((N, K))
        for k in range(K):
            log_prob[:, k] = np.log(pi[k] + 1e-12) + _multivariate_normal_logpdf(X, means[k], covs[k], reg_covar)
            
        # Log-sum-exp trick for responsibilities
        max_log_prob = np.max(log_prob, axis=1, keepdims=True)
        log_resp_unnorm = log_prob - max_log_prob
        resp_unnorm = np.exp(log_resp_unnorm)
        sum_resp = np.sum(resp_unnorm, axis=1, keepdims=True)
        
        loglik = np.sum(max_log_prob.squeeze() + np.log(sum_resp.squeeze()))
        r = resp_unnorm / sum_resp  # (N, K)
        
        # --- M Step ---
        N_k = np.sum(r, axis=0) + 1e-10  # Prevent division by zero
        pi = N_k / N
        
        # Update means
        for k in range(K):
            means[k] = np.dot(r[:, k], X) / N_k[k]
            
        # Compute empirical covariance matrices S_k for each cluster
        Sk_list = []
        for k in range(K):
            diff = X - means[k]  # (N, D)
            # Weighted outer product
            Sk = np.dot((r[:, k, None] * diff).T, diff) / N_k[k]
            Sk_list.append(Sk)
            
        # Update covariances based on structure
        if cov_type == 'full':
            for k in range(K):
                covs[k] = Sk_list[k] + np.eye(D) * 1e-6
        elif cov_type == 'tied':
            tied_cov = np.zeros((D, D))
            for k in range(K):
                tied_cov += (N_k[k] / N) * Sk_list[k]
            tied_cov += np.eye(D) * 1e-6
            for k in range(K):
                covs[k] = np.copy(tied_cov)
        elif cov_type == 'diag':
            for k in range(K):
                covs[k] = np.diag(np.diag(Sk_list[k])) + np.eye(D) * 1e-6
        elif cov_type == 'spherical':
            for k in range(K):
                tr_val = np.trace(Sk_list[k]) / D
                covs[k] = tr_val * np.eye(D) + np.eye(D) * 1e-6

    # Final labels: argmax responsibilities, tie-breaking by smaller k
    final_log_prob = np.zeros((N, K))
    for k in range(K):
        final_log_prob[:, k] = np.log(pi[k] + 1e-12) + _multivariate_normal_logpdf(X, means[k], covs[k], reg_covar)
    labels = np.argmax(final_log_prob, axis=1).tolist()
    
    # Final total log-likelihood
    max_lp = np.max(final_log_prob, axis=1, keepdims=True)
    total_loglik = float(np.sum(max_lp.squeeze() + np.log(np.sum(np.exp(final_log_prob - max_lp), axis=1))))

    return pi, means, covs, labels, total_loglik


def n_params(K, D, cov_type):
    """
    Calculate the number of free parameters in the GMM.
    """
    mean_weight_params = K * D + (K - 1)
    
    if cov_type == 'full':
        cov_params = K * (D * (D + 1)) // 2
    elif cov_type == 'tied':
        cov_params = (D * (D + 1)) // 2
    elif cov_type == 'diag':
        cov_params = K * D
    elif cov_type == 'spherical':
        cov_params = K
    else:
        raise ValueError(f"Unknown covariance type: {cov_type}")
        
    return mean_weight_params + cov_params


def bic(loglik, K, D, N, cov_type):
    """
    Calculate the Bayesian Information Criterion (BIC): log p(D | \hat{\theta}) - 0.5 * p * log N
    """
    p = n_params(K, D, cov_type)
    return loglik - 0.5 * p * np.log(N)