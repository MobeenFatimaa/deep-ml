import math
import numpy as np


def log_softmax(z):
    """Computes the row-wise numerically stable log-softmax of a 2-D array."""
    z = np.asarray(z, dtype=float)
    max_z = np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z - max_z)
    return (z - max_z) - np.log(np.sum(exp_z, axis=1, keepdims=True))


def kl_per_position(logits_ref, logits_q):
    """
    Computes KL divergence KL(p_ref || p_q) for each position (row), 
    where p_ref and p_q are softmax distributions derived from logits.
    """
    logits_ref = np.asarray(logits_ref, dtype=float)
    logits_q = np.asarray(logits_q, dtype=float)
    
    log_p_ref = log_softmax(logits_ref)
    log_p_q = log_softmax(logits_q)
    p_ref = np.exp(log_p_ref)
    
    # KL(p_ref || p_q) = sum_v p_ref(v) * (log p_ref(v) - log p_q(v))
    kl = np.sum(p_ref * (log_p_ref - log_p_q), axis=1)
    return kl


def top1_agreement(logits_ref, logits_q):
    """
    Computes the fraction of rows whose argmax prediction matches between ref and q.
    """
    logits_ref = np.asarray(logits_ref, dtype=float)
    logits_q = np.asarray(logits_q, dtype=float)
    
    argmax_ref = np.argmax(logits_ref, axis=1)
    argmax_q = np.argmax(logits_q, axis=1)
    
    return float(np.mean(argmax_ref == argmax_q))


def quality_report(logits_ref, logits_q):
    """
    Returns a quality report dictionary with mean_kl, max_kl, p99_kl, 
    and top1_agreement, each rounded to 6 decimal places.
    """
    kls = kl_per_position(logits_ref, logits_q)
    agreement = top1_agreement(logits_ref, logits_q)
    
    mean_kl = float(np.mean(kls))
    max_kl = float(np.max(kls))
    
    # p99_kl: sorted KL value at one-based index ceil(0.99 * n)
    n = len(kls)
    if n == 0:
        p99_kl = 0.0
    else:
        sorted_kls = np.sort(kls)
        idx = math.ceil(0.99 * n) - 1
        idx = min(max(idx, 0), n - 1)
        p99_kl = float(sorted_kls[idx])
        
    return {
        "mean_kl": round(mean_kl, 6),
        "max_kl": round(max_kl, 6),
        "p99_kl": round(p99_kl, 6),
        "top1_agreement": round(agreement, 6)
    }


def passes_gate(report, max_mean_kl, min_agreement):
    """
    Returns whether mean_kl <= max_mean_kl and top1_agreement >= min_agreement.
    """
    return report["mean_kl"] <= max_mean_kl and report["top1_agreement"] >= min_agreement