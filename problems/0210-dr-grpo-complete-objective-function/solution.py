import numpy as np

def compute_dr_grpo_objective(log_probs_new: list[list[float]], 
                               log_probs_old: list[list[float]], 
                               rewards: list[float], 
                               epsilon: float = 0.2) -> float:
    """
    Compute the Dr. GRPO (GRPO Done Right) clipped objective.
    
    Args:
        log_probs_new: Log probabilities from new policy π_θ
                    Each response: [log π_θ(o_1|q), log π_θ(o_2|q,o_1), ...]
        log_probs_old: Log probabilities from old policy π_θ_old
        rewards: Rewards R(q, o_i) for each response
        epsilon: Clipping parameter for importance ratios
    
    Returns:
        Dr. GRPO objective value
    """
    G = len(rewards)
    rewards_arr = np.array(rewards, dtype=np.float64)
    
    # 1. Compute Advantage: A_i = R_i - mean(R) (no std normalization, no length normalization)
    mean_reward = np.mean(rewards_arr)
    advantages = rewards_arr - mean_reward
    
    total_objective = 0.0
    
    # 2. Iterate over each response in the group
    for i in range(G):
        lp_new = np.array(log_probs_new[i], dtype=np.float64)
        lp_old = np.array(log_probs_old[i], dtype=np.float64)
        
        # Token ratio: ρ_i,t = exp(log_probs_new - log_probs_old)
        ratios = np.exp(lp_new - lp_old)
        A_i = advantages[i]
        
        # Clipped objective per token
        surr1 = ratios * A_i
        surr2 = np.clip(ratios, 1.0 - epsilon, 1.0 + epsilon) * A_i
        token_objectives = np.minimum(surr1, surr2)
        
        # Sum (not average) over tokens within each response
        total_objective += np.sum(token_objectives)
        
    # 3. Outer group average (1 / G)
    dr_grpo_obj = total_objective / G
    
    return float(dr_grpo_obj)