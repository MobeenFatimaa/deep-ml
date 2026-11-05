import numpy as np


def dormant_ratio(H, tau):
    """(fraction of dormant units, sorted dormant indices) using scores relative to the layer mean."""
    mean_abs = np.abs(H).mean(axis=0)
    mean_of_means = mean_abs.mean()
    
    if mean_of_means == 0:
        scores = np.zeros_like(mean_abs)
    else:
        scores = mean_abs / mean_of_means
        
    dormant_mask = scores <= tau
    dormant_indices = np.where(dormant_mask)[0].tolist()
    ratio = float(len(dormant_indices) / H.shape[1])
    
    return ratio, dormant_indices


def effective_rank(H, eps=1e-12):
    """exp(entropy of normalised singular values) of the column-centred activations."""
    Hc = H - H.mean(axis=0)
    sigma = np.linalg.svd(Hc, compute_uv=False)
    
    sigma_filtered = sigma[sigma >= eps]
    if len(sigma_filtered) == 0:
        return 0.0
        
    total = np.sum(sigma_filtered)
    if total == 0:
        return 0.0
        
    p = sigma_filtered / total
    entropy = -np.sum(p * np.log(p))
    
    return float(np.exp(entropy))


def plasticity_report(H, W, tau, decimals=4):
    """Return a dict with dormant_ratio, effective_rank, and weight_norm rounded to decimals."""
    ratio, _ = dormant_ratio(H, tau)
    erank = effective_rank(H)
    wnorm = np.linalg.norm(W)
    
    return {
        "dormant_ratio": round(ratio, decimals),
        "effective_rank": round(erank, decimals),
        "weight_norm": round(float(wnorm), decimals),
    }