import numpy as np

def cross_entropy_derivative(logits, target):
    logits = np.asarray(logits, dtype=np.float64)
    
    # Numerically stable softmax: p_i = exp(z_i - max(z)) / sum(exp(z - max(z)))
    shift_logits = logits - np.max(logits)
    exps = np.exp(shift_logits)
    p = exps / np.sum(exps)
    
    # Gradient of Cross-Entropy w.r.t. Logits: p - y
    grad = p.copy()
    grad[target] -= 1.0
    
    return grad