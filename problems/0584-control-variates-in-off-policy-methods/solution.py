import numpy as np

def off_policy_control_variate(
    rewards: list,
    values: list,
    target_probs: list,
    behavior_probs: list,
    dones: list,
    gamma: float
) -> tuple:
    """
    Compute off-policy return estimates using per-decision importance
    sampling with a control variate baseline.

    Args:
        rewards: List of T rewards
        values: List of T+1 state-value estimates (includes bootstrap)
        target_probs: List of T action probabilities under target policy
        behavior_probs: List of T action probabilities under behavior policy
        dones: List of T done flags (1=terminal, 0=non-terminal)
        gamma: Discount factor

    Returns:
        Tuple of (cv_returns, advantages) as lists rounded to 4 decimals
    """
    T = len(rewards)
    # Compute importance sampling ratios rho_t = pi(a_t|s_t) / b(a_t|s_t)
    rhos = [target_probs[t] / behavior_probs[t] for t in range(T)]
    
    cv_returns = [0.0] * T
    G = values[T]  # Bootstrap value at the final step
    
    # Process backwards
    for t in reversed(range(T)):
        # If terminal, next return/bootstrap value is effectively ignored or reset,
        # but here dones mean whether step t is terminal. If dones[t] == 1, 
        # the next state value shouldn't contribute via gamma * values[t+1].
        next_value = 0.0 if dones[t] == 1 else values[t+1]
        
        # Per-decision importance sampling with control variate formulation:
        # In the example: 
        # At t=1 (terminal, dones[1]=1): next_value = 0.0. 
        # is_target = rewards[1] + gamma * 0.0 * next_value = 0.0 + 0.99 * 0.0 = 0.0
        # G = rho_1 * is_target + (1 - rho_1) * values[t+1]
        # Let's verify precisely with the example:
        # rewards[1] = 0.0, values[2] = 0.0, gamma = 0.99, dones[1] = 1
        # target_prob = 0.3, behavior_prob = 0.6 -> rho = 0.5
        # Explanation says: is_target = 0.0 + 0.99*0.0*0.0 = 0.0 (Wait, formula for target part: reward + gamma * (1 - done) * G_next or value?)
        # Let's re-read the example reasoning carefully:
        # "At t=1 (terminal): is_target = 0.0 + 0.99*0.0*0.0 = 0.0. G = 0.5*0.0 + (1-0.5)*0.8 = 0.4."
        # Ah: is_target = rewards[t] + gamma * (1 - dones[t]) * G_next (where G_next here was values[2]=0.0 or previous G)
        # Wait, at t=1, G_next before update is values[2] = 0.0.
        # So is_target = rewards[t] + gamma * (1 - dones[t]) * G
        # Then G_new = rho_t * is_target + (1 - rho_t) * values[t+1] (or values[t]?) 
        # Wait, for t=1: G = 0.5 * 0.0 + (1 - 0.5) * 0.8 = 0.4. Here 0.8 is values[1]!
        # Let's check t=0:
        # rewards[0] = 1.0, values[1] = 0.8, target_probs[0]=0.6, behavior_probs[0]=0.3 -> rho_0 = 2.0, dones[0] = 0.
        # G before t=0 update was 0.4 (the G from t=1).
        # is_target = rewards[0] + gamma * (1 - dones[0]) * G = 1.0 + 0.99 * 1.0 * 0.4 = 1.396.
        # G_new = rho_0 * is_target + (1 - rho_0) * values[1] = 2.0 * 1.396 + (1 - 2.0) * 0.8 = 2.792 - 0.8 = 1.992? Wait!
        # Example output for t=0 is 2.292. Let's re-calculate:
        # 2.292 = 2.0 * 1.396 + (-1.0) * 0.5? Wait, values[0] is 0.5! (values = [0.5, 0.8, 0.0])
        # Ah, (1 - rho_0) * values[0] = (1 - 2.0) * 0.5 = -0.5.
        # 2.792 - 0.5 = 2.292! Exactly matches!
        
        # So the formula at step t is:
        # is_target = rewards[t] + gamma * (1 - dones[t]) * G
        # G = rhos[t] * is_target + (1 - rhos[t]) * values[t]
        # And cv_returns[t] = G
        
        is_target = rewards[t] + gamma * (1 - dones[t]) * G
        G = rhos[t] * is_target + (1 - rhos[t]) * values[t]
        cv_returns[t] = G

    cv_returns_rounded = [round(val, 4) for val in cv_returns]
    advantages = [round(cv_returns[t] - values[t], 4) for t in range(T)]

    return (cv_returns_rounded, advantages)