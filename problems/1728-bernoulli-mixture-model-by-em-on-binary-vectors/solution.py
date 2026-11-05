import numpy as np

def bmm_em(X, pi0, mu0, iters, a):
    """
    Fits a Bernoulli Mixture Model (BMM) using Expectation-Maximization (EM).
    
    Args:
        X: binary array of shape (N, D)
        pi0: initial cluster weights of shape (K,)
        mu0: initial bit probabilities of shape (K, D) strictly inside (0, 1)
        iters: number of EM iterations to run
        a: pseudo-count for smoothing (Beta prior parameter)
        
    Returns:
        (pi, mu, loglik_list): final cluster weights, final bit probabilities, and list of log-likelihoods
    """
    N, D = X.shape
    K = pi0.shape[0]
    
    pi = pi0.copy()
    mu = mu0.copy()
    
    loglik_list = []
    
    for _ in range(iters):
        # --- E-step ---
        # Compute log probabilities in log space to avoid numerical underflow for large D.
        # p(x_n | z_n = k) = prod_d (mu_kd)^x_nd * (1 - mu_kd)^(1 - x_nd)
        # log p(x_n | z_n = k) = sum_d [ x_nd * log(mu_kd) + (1 - x_nd) * log(1 - mu_kd) ]
        
        # Clip mu to avoid log(0)
        mu = np.clip(mu, 1e-15, 1.0 - 1e-15)
        
        log_mu = np.log(mu)            # (K, D)
        log_one_minus_mu = np.log(1.0 - mu)  # (K, D)
        
        # X is (N, D), log_mu is (K, D). 
        # term1: X @ log_mu.T -> (N, K)
        term1 = X @ log_mu.T
        # (1 - X) @ log_one_minus_mu.T -> (N, K)
        term2 = (1.0 - X) @ log_one_minus_mu.T
        
        log_p_x_given_z = term1 + term2  # (N, K)
        
        # log(pi) shape (K,)
        log_pi = np.log(np.clip(pi, 1e-15, 1.0))
        
        # log joint: log p(x_n, z_n = k) = log pi_k + log p(x_n | z_n = k)
        log_joint = log_p_x_given_z + log_pi  # (N, K)
        
        # Log-sum-exp trick for marginal log-likelihood: log p(x_n) = log sum_k exp(log_joint_nk)
        max_log_joint = np.max(log_joint, axis=1, keepdims=True)  # (N, 1)
        log_p_x = max_log_joint + np.log(np.sum(np.exp(log_joint - max_log_joint), axis=1, keepdims=True))  # (N, 1)
        
        total_loglik = float(np.sum(log_p_x))
        loglik_list.append(total_loglik)
        
        # Responsibilities: log r_nk = log_joint_nk - log p(x_n)
        log_r = log_joint - log_p_x  # (N, K)
        r = np.exp(log_r)  # (N, K)
        
        # --- M-step ---
        # N_k = sum_n r_nk
        N_k = np.sum(r, axis=0)  # (K,)
        
        # Update pi_k = N_k / N
        pi = N_k / N
        
        # Update mu_kd with pseudo-counts:
        # numerator: sum_n r_nk * x_nd + a
        # denominator: N_k + 2a
        numerator = r.T @ X  # (K, D) + a implicitly across rows via broadcasting
        numerator = numerator + a
        denominator = N_k[:, np.newaxis] + 2.0 * a  # (K, 1)
        
        mu = numerator / denominator
        
    return pi, mu, loglik_list