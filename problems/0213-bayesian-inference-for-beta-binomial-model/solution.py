import numpy as np


def bayesian_inference_beta_binomial(
    prior_alpha: float, prior_beta: float, successes: int, trials: int
) -> tuple[float, float, float]:
    """
    Perform Bayesian inference for Beta-Binomial model.

    Args:
        prior_alpha: Alpha parameter of Beta prior
        prior_beta: Beta parameter of Beta prior
        successes: Number of successes observed
        trials: Total number of trials

    Returns:
        Tuple of (posterior_alpha, posterior_beta, posterior_mean) where:
        - posterior_alpha: Updated alpha parameter
        - posterior_beta: Updated beta parameter
        - posterior_mean: Mean of posterior distribution
    """
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("prior_alpha and prior_beta must be positive.")
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("Invalid successes or trials count.")

    failures = trials - successes
    posterior_alpha = float(prior_alpha + successes)
    posterior_beta = float(prior_beta + failures)
    posterior_mean = float(np.round(posterior_alpha / (posterior_alpha + posterior_beta), 4))

    return (posterior_alpha, posterior_beta, posterior_mean)