import numpy as np

def multivariate_kl_divergence(mu_p: np.ndarray, Cov_p: np.ndarray, mu_q: np.ndarray, Cov_q: np.ndarray) -> float:
    """
    Computes the KL divergence between two multivariate Gaussian distributions.
    
    Parameters:
    mu_p: mean vector of the first distribution
    Cov_p: covariance matrix of the first distribution
    mu_q: mean vector of the second distribution
    Cov_q: covariance matrix of the second distribution

    Returns:
    KL divergence as a float
    """
    mu_p = np.asarray(mu_p, dtype=float)
    Cov_p = np.asarray(Cov_p, dtype=float)
    mu_q = np.asarray(mu_q, dtype=float)
    Cov_q = np.asarray(Cov_q, dtype=float)
    
    k = mu_p.shape[0]
    
    # Compute inverse of covariance matrix Q for quadratic forms and traces
    Cov_q_inv = np.linalg.inv(Cov_q)
    
    # 1. Trace term: tr(Cov_q^{-1} Cov_p)
    trace_term = np.trace(np.dot(Cov_q_inv, Cov_p))
    
    # 2. Mahalanobis distance term: (mu_q - mu_p)^T Cov_q^{-1} (mu_q - mu_p)
    diff = mu_q - mu_p
    mahalanobis_term = np.dot(diff.T, np.dot(Cov_q_inv, diff))
    
    # 3. Log-determinant term: ln(|Cov_q| / |Cov_p|) using slogdet for numerical stability
    _, logdet_q = np.linalg.slogdet(Cov_q)
    _, logdet_p = np.linalg.slogdet(Cov_p)
    logdet_term = logdet_q - logdet_p
    
    # Combine terms using the standard KL divergence formula
    kl_div = 0.5 * (trace_term + mahalanobis_term - k + logdet_term)
    
    return float(kl_div)