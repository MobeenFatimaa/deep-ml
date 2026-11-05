import numpy as np
from math import lgamma, log, pi, exp, sqrt

def log_beta(a, b):
    return lgamma(a) + lgamma(b) - lgamma(a + b)

def log_p_theta(theta, N1, N0, a, b):
    if theta <= 0 or theta >= 1:
        return -float('inf')
    return (N1 + a - 1) * log(theta) + (N0 + b - 1) * log(1 - theta) - log_beta(a, b)

def grid_posterior(N1, N0, a, b, K):
    """Compute the grid approximation of the posterior over K midpoints."""
    grid = np.array([(k + 0.5) / K for k in range(K)])
    log_masses = np.array([log_p_theta(th, N1, N0, a, b) for th in grid])
    max_log = np.max(log_masses)
    masses = np.exp(log_masses - max_log)
    masses /= np.sum(masses)
    return grid, masses

def laplace_theta(N1, N0, a, b):
    """Laplace approximation in theta space."""
    mode = (N1 + a - 1) / (N1 + N0 + a + b - 2)
    H = (N1 + a - 1) / (mode ** 2) + (N0 + b - 1) / ((1 - mode) ** 2)
    var = 1.0 / H
    
    E_mode = -log_p_theta(mode, N1, N0, a, b)
    log_Z = -E_mode + 0.5 * log(2 * pi) - 0.5 * log(H)
    return mode, var, log_Z

def laplace_logit(N1, N0, a, b):
    """Laplace approximation in alpha = logit(theta) space."""
    theta_mode = (N1 + a) / (N1 + N0 + a + b)
    mode = log(theta_mode / (1 - theta_mode))
    
    H = (N1 + N0 + a + b) * theta_mode * (1 - theta_mode)
    var = 1.0 / H
    
    E_mode = - (N1 + a) * mode + (N1 + N0 + a + b) * log(1 + exp(mode)) - log_beta(a, b)
    log_Z = -E_mode + 0.5 * log(2 * pi) - 0.5 * log(H)
    return mode, var, log_Z

def laplace_tv(N1, N0, a, b, space, K=2000):
    """Total variation distance between exact posterior and Laplace approximation."""
    grid = np.array([(k + 0.5) / K for k in range(K)])
    
    A = a + N1
    B = b + N0
    log_B_post = log_beta(A, B)
    
    p_vals = np.array([exp((A - 1) * log(th) + (B - 1) * log(1 - th) - log_B_post) for th in grid])
    
    if space == "theta":
        mode, var, _ = laplace_theta(N1, N0, a, b)
        std = sqrt(var)
        q_vals = np.array([exp(-0.5 * ((th - mode) / std) ** 2) / (std * sqrt(2 * pi)) for th in grid])
    elif space == "logit":
        mode, var, _ = laplace_logit(N1, N0, a, b)
        std = sqrt(var)
        q_vals = np.array([(exp(-0.5 * ((log(th / (1 - th)) - mode) / std) ** 2) / (std * sqrt(2 * pi))) / (th * (1 - th)) for th in grid])
    else:
        q_vals = np.zeros_like(p_vals)
        
    tv = 0.5 * np.sum(np.abs(p_vals - q_vals)) / K
    return tv