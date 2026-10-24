import numpy as np

def compute_elbo(x: list[float], q_mean: float, q_std: float, 
                 prior_mean: float, prior_std: float,
                 likelihood_std: float, n_samples: int = 1000) -> float:
    """
    Compute the Evidence Lower Bound (ELBO) for variational inference.
    
    Args:
        x: Observed data points
        q_mean: Mean of variational distribution q(z)
        q_std: Standard deviation of variational distribution q(z)
        prior_mean: Mean of prior distribution p(z)
        prior_std: Standard deviation of prior distribution p(z)
        likelihood_std: Standard deviation of likelihood p(x|z)
        n_samples: Number of Monte Carlo samples
    
    Returns:
        ELBO value (float)
    """
    # Draw n_samples values of z from q(z)
    z_samples = np.random.normal(q_mean, q_std, n_samples)
    
    # Compute expected log-likelihood: E_q [ sum_i log p(x_i | z) ]
    x_arr = np.array(x)
    # Shape of diff: (n_samples, len(x))
    diff = x_arr[None, :] - z_samples[:, None]
    log_likelihood_terms = -0.5 * np.log(2 * np.pi * likelihood_std**2) - (diff ** 2) / (2 * likelihood_std**2)
    expected_log_likelihood = np.mean(np.sum(log_likelihood_terms, axis=1))
    
    # Compute expected log-prior: E_q [ log p(z) ]
    log_prior_terms = -0.5 * np.log(2 * np.pi * prior_std**2) - ((z_samples - prior_mean) ** 2) / (2 * prior_std**2)
    expected_log_prior = np.mean(log_prior_terms)
    
    # Compute closed-form entropy of q(z): H[q] = 1/2 * log(2 * pi * e * q_std^2)
    entropy = 0.5 * np.log(2 * np.pi * np.e * q_std**2)
    
    # Total ELBO
    elbo = expected_log_likelihood + expected_log_prior + entropy
    
    return float(elbo)