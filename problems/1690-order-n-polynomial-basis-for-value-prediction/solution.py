import numpy as np

def polynomial_basis(s, n):
    """
    Returns the order-n polynomial basis feature vector for a state s.
    """
    s = np.asarray(s, dtype=float)
    k = len(s)
    features = []
    # np.ndindex generates exponent vectors lexicographically
    # with the last dimension varying fastest (base n+1 counting)
    for c in np.ndindex(*((n + 1,) * k)):
        feat = np.prod(s ** np.array(c, dtype=float))
        features.append(feat)
    return np.array(features, dtype=float)

def gradient_mc_poly(samples, n, alpha, n_passes):
    """
    Fits a linear value function using gradient Monte Carlo updates over n_passes.
    """
    if not samples:
        return np.array([])
    
    # Determine the dimension k from the first sample's state
    k = len(samples[0][0])
    num_features = (n + 1) ** k
    
    # Start from w = 0
    w = np.zeros(num_features, dtype=float)
    
    for _ in range(n_passes):
        for s, G in samples:
            x = polynomial_basis(s, n)
            v_hat = np.dot(w, x)
            w = w + alpha * (G - v_hat) * x
            
    return w