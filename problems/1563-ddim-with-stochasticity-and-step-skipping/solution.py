import numpy as np


def ddim_timesteps(T, S, spacing="linear"):
    """
    S increasing integer timesteps in [0, T-1], starting at 0 and ending at T-1.
    "linear": np.linspace(0, T-1, S) rounded; "quadratic": np.linspace(0, sqrt(T-1), S)**2 rounded.
    Returns a Python list of ints. Raise ValueError for other spacings.
    """
    if spacing == "linear":
        steps = np.linspace(0, T - 1, S)
    elif spacing == "quadratic":
        steps = np.linspace(0, np.sqrt(T - 1), S) ** 2
    else:
        raise ValueError(f"Unsupported spacing type: {spacing}")
        
    return [int(round(s)) for s in steps]


def ddim_sigma(a_t, a_prev, eta):
    """
    sigma = eta * sqrt((1 - a_prev) / (1 - a_t)) * sqrt(1 - a_t / a_prev)
    """
    if eta == 0.0 or a_prev >= 1.0 or a_t >= a_prev:
        return 0.0
    term1 = (1.0 - a_prev) / (1.0 - a_t)
    term2 = 1.0 - (a_t / a_prev)
    if term1 < 0 or term2 < 0:
        return 0.0
    return float(eta * np.sqrt(term1) * np.sqrt(term2))


def ddim_step(x_t, eps, a_t, a_prev, eta, rng=None):
    """
    General DDIM update (Song et al. 2021, eq. 12).
    Draw z = rng.standard_normal(x_t.shape) only when sigma > 0.
    """
    # Predict x0
    x0_pred = (x_t - np.sqrt(1.0 - a_t) * eps) / np.sqrt(a_t)
    
    # Calculate sigma
    sigma = ddim_sigma(a_t, a_prev, eta)
    
    # Direction pointing to x_t
    dir_xt = np.sqrt(max(1.0 - a_prev - sigma ** 2, 0.0)) * eps
    
    # Stochastic noise term z
    if sigma > 0:
        if rng is None:
            raise ValueError("An RNG must be provided when sigma > 0 and eta > 0.")
        z = rng.standard_normal(x_t.shape)
    else:
        z = np.zeros_like(x_t)
        
    x_prev = np.sqrt(a_prev) * x0_pred + dir_xt + sigma * z
    return x_prev


def ddim_sample(x_T, eps_model, alpha_bar, timesteps, eta, rng=None):
    """
    Run ddim_step over the timesteps from largest to smallest.
    a_t = alpha_bar[t]; a_prev = alpha_bar[next smaller timestep], or 1.0 after the smallest one.
    eps_model(x, t) is called exactly once per timestep. Returns the final array.
    """
    x = np.copy(x_T)
    
    # Sort timesteps in descending order (from T down to 0)
    sorted_steps = sorted(timesteps, reverse=True)
    
    for i, t in enumerate(sorted_steps):
        eps = eps_model(x, t)
        a_t = alpha_bar[t]
        
        # a_prev is alpha_bar[next smaller timestep], or 1.0 after the smallest timestep
        if i < len(sorted_steps) - 1:
            t_prev = sorted_steps[i + 1]
            a_prev = alpha_bar[t_prev]
        else:
            a_prev = 1.0
            
        x = ddim_step(x, eps, a_t, a_prev, eta, rng=rng)
        
    return x