import numpy as np


def streaming_frequency(batches, alpha):
    A = {}
    B = {}
    for t, batch in enumerate(batches, 1):
        # Handle duplicate items within the same batch or process unique items per step
        seen_in_batch = set(batch)
        for item in seen_in_batch:
            if item not in A:
                A[item] = float(t)
                B[item] = t
            else:
                gap = t - B[item]
                A[item] = (1 - alpha) * A[item] + alpha * gap
                B[item] = t
                
    result = {}
    for item in A:
        prob = 1.0 / A[item]
        result[item] = round(float(prob), 4)
    return result


def logq_corrected_logits(scores, item_ids, probs):
    scores = np.array(scores, dtype=float)
    log_probs = np.array([np.log(probs[item_id]) for item_id in item_ids])
    # Subtract log(q_j) from each column j
    corrected = scores - log_probs[None, :]
    return np.round(corrected, 4)


def in_batch_softmax_loss(logits):
    logits = np.array(logits, dtype=float)
    # Numerically stable log-softmax per row
    max_vals = np.max(logits, axis=-1, keepdims=True)
    exp_shifted = np.exp(logits - max_vals)
    log_sum_exp = max_vals + np.log(np.sum(exp_shifted, axis=-1, keepdims=True))
    log_softmax = logits - log_sum_exp
    
    # Positive is at column i for row i
    b = logits.shape[0]
    losses = -log_softmax[np.arange(b), np.arange(b)]
    return round(float(np.mean(losses)), 4)


def correction_effect(scores, item_ids, probs):
    scores = np.array(scores, dtype=float)
    uncorrected_loss = in_batch_softmax_loss(scores)
    
    corrected_logits = logq_corrected_logits(scores, item_ids, probs)
    corrected_loss = in_batch_softmax_loss(corrected_logits)
    
    return uncorrected_loss, corrected_loss