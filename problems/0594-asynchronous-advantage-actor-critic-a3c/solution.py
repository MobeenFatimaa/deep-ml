import numpy as np

def a3c_update(
    worker_trajectories: list,
    policy_params: list,
    value_params: list,
    n_steps: int,
    gamma: float,
    alpha_policy: float,
    alpha_value: float,
    entropy_coeff: float
) -> dict:
    """
    Simulate the A3C update across multiple workers.

    Args:
        worker_trajectories: List of episodes, each a list of (state, action, reward) tuples.
        policy_params: 2D list of shape (num_states, num_actions) - softmax logits.
        value_params: 1D list of shape (num_states,) - tabular value estimates.
        n_steps: Number of steps for n-step returns.
        gamma: Discount factor.
        alpha_policy: Policy learning rate.
        alpha_value: Value function learning rate.
        entropy_coeff: Entropy regularization coefficient.

    Returns:
        Dictionary with 'policy_params' and 'value_params'.
    """
    policy = np.array(policy_params, dtype=float)
    value = np.array(value_params, dtype=float)
    
    for trajectory in worker_trajectories:
        T = len(trajectory)
        if T == 0:
            continue
            
        policy_grad = np.zeros_like(policy)
        value_grad = np.zeros_like(value)
        
        for t in range(T):
            # Compute n-step return
            end = min(t + n_steps, T)
            R_t = 0.0
            for i in range(t, end):
                R_t += (gamma ** (i - t)) * trajectory[i][2]
                
            if end < T:
                next_state = trajectory[end][0]
                R_t += (gamma ** (end - t)) * value[next_state]
                
            state, action, reward = trajectory[t]
            
            # Compute advantage
            A_t = R_t - value[state]
            
            # Compute softmax policy probabilities
            logits = policy[state]
            exp_logits = np.exp(logits - np.max(logits))
            pi = exp_logits / np.sum(exp_logits)
            
            # Compute policy gradient
            num_actions = policy.shape[1]
            one_hot = np.zeros(num_actions)
            one_hot[action] = 1.0
            
            d_policy = A_t * (one_hot - pi)
            
            # Add entropy regularization if coefficient is non-zero
            if entropy_coeff > 0.0:
                log_pi = np.log(pi + 1e-10)
                entropy = -np.sum(pi * log_pi)
                d_entropy = pi * (-log_pi - entropy)
                d_policy += entropy_coeff * d_entropy
                
            policy_grad[state] += d_policy
            
            # Compute value function gradient
            value_grad[state] += A_t
            
        # Apply accumulated gradients to shared parameters for this worker
        policy += alpha_policy * policy_grad
        value += alpha_value * value_grad
        
    return {
        'policy_params': np.round(policy, 4).tolist(),
        'value_params': np.round(value, 4).tolist()
    }