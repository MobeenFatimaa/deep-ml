import numpy as np


def ddpm_posterior(x0, xt, t, betas):
    """
    Mean and variance of q(x_{t-1} | x_t, x_0) (Ho et al. 2020, eqs. 6-7).

    x0, xt: arrays of the same shape
    t:      1-indexed timestep in 1..T
    betas:  1-D array of length T (betas[t-1] is beta_t)

    Returns (mean, var) with mean an array like x0 and var a Python float.
    """
    x0 = np.asarray(x0, dtype=float)
    xt = np.asarray(xt, dtype=float)
    betas = np.asarray(betas, dtype=float)
    
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    
    alpha_bar_t = alpha_bars[t - 1]
    alpha_bar_t_minus_1 = alpha_bars[t - 2] if t > 1 else 1.0
    beta_t = betas[t - 1]
    alpha_t = alphas[t - 1]
    
    # Posterior variance \tilde{\beta}_t
    var = float((1.0 - alpha_bar_t_minus_1) / (1.0 - alpha_bar_t) * beta_t)
    
    # Posterior mean coefficients
    c0 = np.sqrt(alpha_bar_t_minus_1) * beta_t / (1.0 - alpha_bar_t)
    c1 = np.sqrt(alpha_t) * (1.0 - alpha_bar_t_minus_1) / (1.0 - alpha_bar_t)
    mean = c0 * x0 + c1 * xt
    
    return mean, var


def per_step_kl(x0, xt, t, betas, model_mean, decimals=6):
    """
    The L_{t-1} term: KL between the forward posterior and a model Gaussian with
    mean model_mean and variance tilde_beta_t, plus the equivalent weighted epsilon-MSE.
    Requires t >= 2.

    Returns a dict with keys "posterior_mean", "posterior_var", "kl", "eps_true",
    "eps_model", "weighted_eps_mse", all rounded to `decimals`.
    """
    x0 = np.asarray(x0, dtype=float)
    xt = np.asarray(xt, dtype=float)
    model_mean = np.asarray(model_mean, dtype=float)
    betas = np.asarray(betas, dtype=float)
    
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    
    alpha_bar_t = alpha_bars[t - 1]
    beta_t = betas[t - 1]
    alpha_t = alphas[t - 1]
    
    post_mean, post_var = ddpm_posterior(x0, xt, t, betas)
    
    # KL divergence between forward posterior and model Gaussian (summed over dimensions)
    kl = float(np.sum((post_mean - model_mean) ** 2) / (2.0 * post_var))
    
    # True noise component epsilon
    eps_true = (xt - np.sqrt(alpha_bar_t) * x0) / np.sqrt(1.0 - alpha_bar_t)
    
    # Model's implied noise estimate epsilon_theta derived from model_mean
    eps_model = (np.sqrt(1.0 - alpha_bar_t) / beta_t) * (xt - np.sqrt(alpha_t) * model_mean)
    
    # Equivalent weighted epsilon MSE loss
    weight = (beta_t ** 2) / (2.0 * post_var * alpha_t * (1.0 - alpha_bar_t))
    weighted_eps_mse = float(weight * np.sum((eps_true - eps_model) ** 2))
    
    return {
        "posterior_mean": np.round(post_mean, decimals).tolist(),
        "posterior_var": round(float(post_var), decimals),
        "kl": round(float(kl), decimals),
        "eps_true": np.round(eps_true, decimals).tolist(),
        "eps_model": np.round(eps_model, decimals).tolist(),
        "weighted_eps_mse": round(float(weighted_eps_mse), decimals)
    }