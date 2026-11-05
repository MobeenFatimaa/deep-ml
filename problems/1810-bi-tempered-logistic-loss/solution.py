import numpy as np


def log_t(x, t):
    """Computes the t-logarithm element-wise."""
    x = np.asarray(x, dtype=float)
    if t == 1.0:
        return np.log(x)
    else:
        res = np.zeros_like(x)
        mask = x > 0
        res[mask] = (x[mask] ** (1.0 - t) - 1.0) / (1.0 - t)
        return res


def exp_t(x, t):
    """Computes the t-exponential element-wise with [.]+ clip."""
    x = np.asarray(x, dtype=float)
    if t == 1.0:
        return np.exp(x)
    else:
        base = 1.0 + (1.0 - t) * x
        clipped = np.maximum(0.0, base)
        return clipped ** (1.0 / (1.0 - t))


def tempered_softmax(a, t):
    """
    Computes the tempered softmax for logits a of shape (N, C) and t >= 1.
    Per-row normalization scalar lambda(a) solves sum_c exp_t(a_c - lambda(a)) = 1.
    """
    a = np.asarray(a, dtype=float)
    N, C = a.shape
    
    if t == 1.0:
        max_a = np.max(a, axis=1, keepdims=True)
        exp_a = np.exp(a - max_a)
        return exp_a / np.sum(exp_a, axis=1, keepdims=True)
    
    y = np.zeros_like(a)
    for i in range(N):
        row = a[i]
        low = np.min(row) - 1000.0
        high = np.max(row) + 1000.0
        
        for _ in range(60):
            mid = 0.5 * (low + high)
            probs = exp_t(row - mid, t)
            total = np.sum(probs)
            if total > 1.0:
                low = mid
            else:
                high = mid
                
        y[i] = exp_t(row - 0.5 * (low + high), t)
        s = np.sum(y[i])
        if s > 0:
            y[i] /= s
        else:
            y[i] = 1.0 / C
            
    return y


def bi_tempered_loss(labels, a, t1, t2):
    """
    Computes the bi-tempered logistic loss for given integer labels and logits a of shape (N, C).
    """
    labels = np.asarray(labels, dtype=int)
    a = np.asarray(a, dtype=float)
    N, C = a.shape
    
    # Special case for standard cross-entropy (t1 = 1, t2 = 1) using log-sum-exp for numerical stability
    if t1 == 1.0 and t2 == 1.0:
        max_a = np.max(a, axis=1, keepdims=True)
        log_sum_exp = max_a.squeeze(-1) + np.log(np.sum(np.exp(a - max_a), axis=1))
        return log_sum_exp - a[np.arange(N), labels]
        
    y_hat = tempered_softmax(a, t2)
    y_true_probs = y_hat[np.arange(N), labels]
    
    term1 = -log_t(y_true_probs, t1)
    
    if t1 == 1.0:
        losses = -np.log(np.clip(y_true_probs, 1e-15, 1.0))
    else:
        power = 2.0 - t1
        sum_p_power = np.sum(y_hat ** power, axis=1)
        term2 = (1.0 - sum_p_power) / (2.0 - t1)
        losses = term1 - term2
        
    return losses