import numpy as np
from math import erf, sqrt

def kth_excluding(v, k, i):
    """k-th largest value of v with entry i removed, as a float."""
    v_list = list(v)
    # Remove the i-th entry
    v_reduced = v_list[:i] + v_list[i+1:]
    # Sort descending to find the k-th largest (1-indexed k)
    v_reduced.sort(reverse=True)
    return float(v_reduced[k-1])

def standard_normal_cdf(z):
    """Standard normal CDF using math.erf."""
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))

def load_estimate(clean, noisy, noise_std, k):
    """Load_i = sum over the batch of Phi((clean_i - kth_excluding(noisy, k, i)) / noise_std_i)."""
    clean = np.array(clean, dtype=float)
    noisy = np.array(noisy, dtype=float)
    noise_std = np.array(noise_std, dtype=float)
    
    batch_size, n = clean.shape
    loads = np.zeros(n, dtype=float)
    
    for b in range(batch_size):
        h = noisy[b]
        c = clean[b]
        s = noise_std[b]
        
        for i in range(n):
            thresh = kth_excluding(h, k, i)
            # Avoid division by zero if standard deviation is exceptionally small
            std_val = s[i] if s[i] > 1e-12 else 1e-12
            z = (c[i] - thresh) / std_val
            loads[i] += standard_normal_cdf(z)
            
    return loads

def load_loss(clean, noisy, noise_std, k, w_load):
    """w_load * CV^2(load), CV^2 = population variance / (mean^2 + 1e-10)."""
    loads = load_estimate(clean, noisy, noise_std, k)
    
    mean_load = np.mean(loads)
    # Population variance (ddof=0)
    var_load = np.var(loads)
    
    cv_squared = var_load / (mean_load**2 + 1e-10)
    loss = w_load * cv_squared
    
    return float(loss)