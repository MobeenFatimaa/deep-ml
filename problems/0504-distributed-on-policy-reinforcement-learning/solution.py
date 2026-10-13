import numpy as np

def distributed_onpolicy_step(worker_data: list, gamma: float, lam: float, clip_eps: float) -> dict:
    """
    Simulate one optimization step of distributed on-policy RL.
    
    Args:
        worker_data: List of dicts with keys 'rewards', 'values', 'dones',
                     'old_log_probs', 'new_log_probs' per worker.
        gamma: Discount factor.
        lam: GAE lambda parameter.
        clip_eps: PPO-style clipping epsilon.
    
    Returns:
        Dictionary with aggregated training statistics.
    """
    all_advantages = []
    all_returns = []
    all_ratios = []
    all_value_preds = []
    per_worker_mean_return = []
    
    for worker in worker_data:
        rewards = worker['rewards']
        values = worker['values']
        dones = worker['dones']
        old_log_probs = worker['old_log_probs']
        new_log_probs = worker['new_log_probs']
        
        T = len(rewards)
        advantages = [0.0] * T
        returns = [0.0] * T
        
        last_gae_lam = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
                
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
            returns[t] = advantages[t] + values[t]
            
        all_advantages.extend(advantages)
        all_returns.extend(returns)
        
        # Calculate worker mean return
        worker_mean_ret = float(np.mean(returns)) if returns else 0.0
        per_worker_mean_return.append(round(worker_mean_ret, 4))
        
        # Compute policy probability ratios: exp(new_log_prob - old_log_prob)
        for old_lp, new_lp in zip(old_log_probs, new_log_probs):
            ratio = np.exp(new_lp - old_lp)
            all_ratios.append(ratio)
            
        # Exclude bootstrap value for value loss (take first T values)
        all_value_preds.extend(values[:T])
        
    num_samples = len(all_advantages)
    if num_samples == 0:
        return {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'clip_fraction': 0.0,
            'mean_advantage_raw': 0.0,
            'std_advantage_raw': 0.0,
            'num_samples': 0,
            'per_worker_mean_return': []
        }
        
    adv_arr = np.array(all_advantages, dtype=float)
    ret_arr = np.array(all_returns, dtype=float)
    val_arr = np.array(all_value_preds, dtype=float)
    ratio_arr = np.array(all_ratios, dtype=float)
    
    mean_adv_raw = float(np.mean(adv_arr))
    std_adv_raw = float(np.std(adv_arr))
    
    # Normalize advantages
    if std_adv_raw < 1e-8:
        norm_advantages = adv_arr - mean_adv_raw
    else:
        norm_advantages = (adv_arr - mean_adv_raw) / std_adv_raw
        
    # PPO clipped surrogate objective
    surr1 = ratio_arr * norm_advantages
    surr2 = np.clip(ratio_arr, 1.0 - clip_eps, 1.0 + clip_eps) * norm_advantages
    policy_loss = float(-np.mean(np.minimum(surr1, surr2)))
    
    # Value loss (Mean Squared Error)
    value_loss = float(np.mean((ret_arr - val_arr) ** 2))
    
    # Clip fraction
    clipped = np.abs(ratio_arr - 1.0) > clip_eps
    clip_fraction = float(np.mean(clipped))
    
    return {
        'policy_loss': round(policy_loss, 4),
        'value_loss': round(value_loss, 4),
        'clip_fraction': round(clip_fraction, 4),
        'mean_advantage_raw': round(mean_adv_raw, 4),
        'std_advantage_raw': round(std_adv_raw, 4),
        'num_samples': num_samples,
        'per_worker_mean_return': per_worker_mean_return
    }