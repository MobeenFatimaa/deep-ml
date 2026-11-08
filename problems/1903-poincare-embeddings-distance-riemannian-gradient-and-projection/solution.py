import numpy as np


def poincare_dist(u, v):
    """Computes the Poincaré distance between vectors u and v."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    
    u_sq_norm = np.sum(u ** 2)
    v_sq_norm = np.sum(v ** 2)
    diff_sq_norm = np.sum((u - v) ** 2)
    
    val = 1.0 + 2.0 * diff_sq_norm / ((1.0 - u_sq_norm) * (1.0 - v_sq_norm))
    # Ensure numerical stability for arcosh
    val = max(1.0, val)
    return float(np.arccosh(val))


def poincare_dist_grad(u, v):
    """Computes the Euclidean gradient of d(u, v) with respect to u."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    
    alpha = 1.0 - np.sum(u ** 2)
    beta = 1.0 - np.sum(v ** 2)
    diff_sq = np.sum((u - v) ** 2)
    
    gamma = 1.0 + 2.0 * diff_sq / (alpha * beta)
    gamma = max(1.0 + 1e-12, gamma)
    
    denom = beta * (gamma ** 2 - 1.0) ** 0.5
    if denom == 0:
        return np.zeros_like(u)
        
    v_sq = np.sum(v ** 2)
    u_dot_v = np.dot(u, v)
    
    term = (v_sq - 2.0 * u_dot_v + 1.0) / (alpha ** 2) * u - v / alpha
    grad = 4.0 / denom * term
    return grad


def poincare_loss(Z, pos_pairs, neg_lists):
    """Computes the mean softmax cross-entropy loss over positive pairs and their negatives."""
    Z = np.asarray(Z, dtype=float)
    total_loss = 0.0
    num_pairs = len(pos_pairs)
    
    if num_pairs == 0:
        return 0.0
        
    for k, (u_idx, v_idx) in enumerate(pos_pairs):
        u = Z[u_idx]
        v = Z[v_idx]
        neg_indices = neg_lists[k]
        
        d_pos = poincare_dist(u, v)
        
        # Compute denominator: exp(-d_pos) + sum_{v' in neg} exp(-d(u, v'))
        neg_distances = [poincare_dist(u, Z[neg_idx]) for neg_idx in neg_indices]
        
        # Use log-sum-exp trick for numerical stability
        logits = [-d_pos] + [-d_neg for d_neg in neg_distances]
        max_logit = max(logits)
        
        exp_sum = sum(np.exp(logit - max_logit) for logit in logits)
        log_sum_exp = max_logit + np.log(exp_sum)
        
        # Loss for this pair: - (-d_pos - log_sum_exp) = d_pos + log_sum_exp
        pair_loss = d_pos + log_sum_exp
        total_loss += pair_loss
        
    return float(total_loss / num_pairs)


def riemannian_sgd_step(Z, egrad, lr, eps=1e-5):
    """Performs a Riemannian SGD step and projects points back inside the Poincaré ball."""
    Z = np.asarray(Z, dtype=float)
    egrad = np.asarray(egrad, dtype=float)
    
    new_Z = np.empty_like(Z)
    
    for i in range(len(Z)):
        z = Z[i]
        g = egrad[i]
        
        z_sq_norm = np.sum(z ** 2)
        # Riemannian gradient scaling factor: ((1 - ||z||^2) / 2)^2
        scale = ((1.0 - z_sq_norm) ** 2) / 4.0
        rgrad = scale * g
        
        # Gradient descent step
        updated_z = z - lr * rgrad
        
        # Projection onto the Poincaré ball {z: ||z|| < 1 - eps}
        updated_sq_norm = np.sum(updated_z ** 2)
        max_norm = 1.0 - eps
        
        if updated_sq_norm >= max_norm ** 2:
            updated_z = updated_z / np.sqrt(updated_sq_norm) * max_norm
            
        new_Z[i] = updated_z
        
    return new_Z