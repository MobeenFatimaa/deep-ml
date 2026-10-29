import numpy as np

def async_ppo_pipeline(worker_batches, theta_init, alpha, gamma_stale, clip_eps, max_staleness):
    """
    Asynchronous PPO learner that consumes batches from workers holding possibly stale policies.

    Args:
        worker_batches: list of dicts with keys 'policy_version', 'states', 'actions', 'old_log_probs', 'advantages'
        theta_init: initial parameters, shape (n_features, n_actions)
        alpha: base learning rate
        gamma_stale: staleness discount factor
        clip_eps: PPO clipping epsilon
        max_staleness: maximum allowed lag (v - policy_version)

    Returns:
        dict with keys 'theta', 'batches_used', 'batches_discarded', 'avg_clip_fraction'
    """
    theta = np.array(theta_init, dtype=float)
    v = 0
    batches_used = 0
    batches_discarded = 0
    clip_fractions = []
    
    for batch in worker_batches:
        policy_version = batch['policy_version']
        l = v - policy_version
        
        if l > max_staleness:
            batches_discarded += 1
            continue
            
        states = np.array(batch['states'], dtype=float)
        actions = np.array(batch['actions'], dtype=int)
        old_log_probs = np.array(batch['old_log_probs'], dtype=float)
        advantages = np.array(batch['advantages'], dtype=float)
        
        N = states.shape[0]
        if N == 0:
            batches_used += 1
            clip_fractions.append(0.0)
            v += 1
            continue
            
        # Forward pass: compute current probabilities using softmax(s * theta)
        logits = states @ theta
        max_logits = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Log probabilities of taken actions
        pi_actions = probs[np.arange(N), actions]
        pi_actions = np.clip(pi_actions, 1e-12, 1.0)
        log_probs = np.log(pi_actions)
        
        # Importance sampling ratios
        ratios = np.exp(log_probs - old_log_probs)
        
        # PPO clipping condition
        clipped_mask = (ratios * advantages) > (np.clip(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages)
        
        clip_fraction = np.mean(clipped_mask)
        clip_fractions.append(clip_fraction)
        
        # Gradient calculation using non-clipped samples only
        not_clipped = ~clipped_mask
        if np.any(not_clipped):
            # Construct one-hot action matrix E of shape (N, n_actions)
            n_actions = probs.shape[1]
            E = np.zeros_like(probs)
            E[np.arange(N), actions] = 1.0
            
            diff = E - probs  # (N, n_actions)
            
            # Weights for samples: A_i * r_i for non-clipped, 0 for clipped
            weights = advantages * ratios
            weights[clipped_mask] = 0.0
            
            weighted_diff = weights[:, None] * diff  # (N, n_actions)
            grad = (states.T @ weighted_diff) / N   # (n_features, n_actions)
            
            # Gradient ascent with staleness-decayed step
            effective_lr = alpha * (gamma_stale ** l)
            theta += effective_lr * grad
            
        batches_used += 1
        v += 1
        
    avg_clip_fraction = float(np.mean(clip_fractions)) if clip_fractions else 0.0
    
    theta_list = [[round(float(val), 4) for val in row] for row in theta.tolist()]
    
    return {
        'theta': theta_list,
        'batches_used': int(batches_used),
        'batches_discarded': int(batches_discarded),
        'avg_clip_fraction': round(float(avg_clip_fraction), 4)
    }