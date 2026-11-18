import numpy as np


def next_byte_entropy(probs):
    """Entropy in bits of each row of an n x V probability array; list of floats."""
    p = np.array(probs, dtype=float)
    # Compute log2(p), treating p == 0 specially (0 * log2(0) = 0)
    log2_p = np.zeros_like(p)
    mask = p > 0
    log2_p[mask] = np.log2(p[mask])
    
    entropies = -np.sum(p * log2_p, axis=1)
    return entropies.tolist()


def entropy_boundaries(entropies, threshold):
    """1 where a chunk starts: byte 0 always, then H_t > threshold."""
    boundaries = []
    for t, h in enumerate(entropies):
        if t == 0:
            boundaries.append(1)
        else:
            if h > threshold:
                boundaries.append(1)
            else:
                boundaries.append(0)
    return boundaries


def bytes_per_chunk(boundaries):
    """Bytes per chunk."""
    n_bytes = len(boundaries)
    n_chunks = sum(boundaries)
    if n_chunks == 0:
        return float(n_bytes)
    return float(n_bytes / n_chunks)


def fit_threshold(entropies, target, iters=60):
    """Bisect the threshold; return hi, the smallest threshold whose compression is at least target."""
    H = list(entropies)
    n = len(H)
    
    if n < 2 or not (1.0 <= target <= float(n)):
        raise ValueError("Number of entropies must be >= 2 and target must be within [1, n].")
    
    lo = min(H[1:]) - 1.0
    hi = max(H[1:])
    
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        bounds = entropy_boundaries(H, mid)
        comp = bytes_per_chunk(bounds)
        
        if comp >= target:
            hi = mid
        else:
            lo = mid
            
    return hi