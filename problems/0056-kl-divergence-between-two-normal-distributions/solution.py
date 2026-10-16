import numpy as np

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):
    """
    Compute the Kullback-Leibler (KL) divergence D_KL(P || Q) between 
    two univariate normal distributions P(mu_p, sigma_p^2) and Q(mu_q, sigma_q^2).
    
    Args:
        mu_p: Mean of distribution P
        sigma_p: Standard deviation of distribution P
        mu_q: Mean of distribution Q
        sigma_q: Standard deviation of distribution Q
        
    Returns:
        KL divergence as a float
    """
    mu_p = float(mu_p)
    sigma_p = float(sigma_p)
    mu_q = float(mu_q)
    sigma_q = float(sigma_q)
    
    # Formula for KL divergence between two normal distributions:
    # D_KL(P || Q) = log(sigma_q / sigma_p) + (sigma_p^2 + (mu_p - mu_q)^2) / (2 * sigma_q^2) - 0.5
    kl = np.log(sigma_q / sigma_p) + (sigma_p**2 + (mu_p - mu_q)**2) / (2 * sigma_q**2) - 0.5
    return float(kl)