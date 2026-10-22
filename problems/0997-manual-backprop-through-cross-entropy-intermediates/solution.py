import numpy as np

def manual_cross_entropy_backward(logits: np.ndarray, y: np.ndarray) -> dict:
    """
    Manually compute gradients through every intermediate of softmax + NLL loss.

    Args:
        logits: numpy array of shape (n, C)
        y:      numpy array of integer class indices, shape (n,)

    Returns:
        Dictionary with keys 'dlogprobs', 'dprobs', 'dcounts_sum_inv',
        'dcounts_sum', 'dcounts', 'dnorm_logits', 'dlogit_maxes', 'dlogits'.
    """
    n, C = logits.shape
    
    # Forward pass to get intermediates needed for backward pass
    logit_maxes = logits.max(axis=1, keepdims=True)
    norm_logits = logits - logit_maxes
    counts = np.exp(norm_logits)
    counts_sum = counts.sum(axis=1, keepdims=True)
    counts_sum_inv = counts_sum ** -1
    probs = counts * counts_sum_inv
    logprobs = np.log(probs)
    
    # 1. dlogprobs
    dlogprobs = np.zeros_like(logprobs)
    dlogprobs[np.arange(n), y] = -1.0 / n
    
    # 2. dprobs
    dprobs = dlogprobs / probs
    
    # 3. dcounts_sum_inv
    dcounts_sum_inv = np.sum(dprobs * counts, axis=1, keepdims=True)
    dcounts_from_probs = dprobs * counts_sum_inv
    
    # 4. dcounts_sum
    dcounts_sum = -dcounts_sum_inv * (counts_sum ** -2)
    dcounts_from_sum = dcounts_sum
    
    # 5. dcounts
    dcounts = dcounts_from_probs + dcounts_from_sum
    
    # 6. dnorm_logits
    dnorm_logits = dcounts * counts
    
    # 7. dlogit_maxes
    # Sum over all columns of -dnorm_logits since norm_logits = logits - logit_maxes
    dlogit_maxes = np.sum(-dnorm_logits, axis=1, keepdims=True)
    
    # 8. dlogits
    # Direct path from norm_logits plus contribution from logit_maxes at the first argmax
    dlogits = dnorm_logits.copy()
    argmax_indices = np.argmax(logits, axis=1)
    dlogits[np.arange(n), argmax_indices] += dlogit_maxes.squeeze(axis=1)
    
    return {
        'dlogprobs': dlogprobs,
        'dprobs': dprobs,
        'dcounts_sum_inv': dcounts_sum_inv,
        'dcounts_sum': dcounts_sum,
        'dcounts': dcounts,
        'dnorm_logits': dnorm_logits,
        'dlogit_maxes': dlogit_maxes,
        'dlogits': dlogits
    }