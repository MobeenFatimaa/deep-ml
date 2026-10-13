import numpy as np

def reinforce_baseline_update(episode: list, theta: np.ndarray, w: np.ndarray, gamma: float, alpha_theta: float, alpha_w: float) -> dict:
    """
    Perform REINFORCE with baseline update for a single episode.
    
    Args:
        episode: list of (state, action, reward) tuples
        theta: policy parameters, shape (n_features, n_actions)
        w: baseline parameters, shape (n_features,)
        gamma: discount factor
        alpha_theta: learning rate for policy
        alpha_w: learning rate for baseline
    
    Returns:
        dict with 'theta', 'w', 'returns', 'advantages'
    """
    T = len(episode)
    
    # 1. Compute discounted returns G_t for each timestep
    returns = [0.0] * T
    G = 0.0
    for t in reversed(range(T)):
        _, _, reward = episode[t]
        G = reward + gamma * G
        returns[t] = G
        
    advantages = []
    
    # Copy parameters to perform sequential updates during the episode
    theta = theta.copy()
    w = w.copy()
    
    for t in range(T):
        state, action, _ = episode[t]
        s = np.array(state, dtype=float)
        G_t = returns[t]
        
        # Baseline prediction: V(s) = w^T @ s
        baseline_pred = float(np.dot(w, s))
        advantage = G_t - baseline_pred
        advantages.append(advantage)
        
        # Softmax policy probabilities with numerical stability
        logits = np.dot(s, theta)
        logits -= np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)
        
        # Gradient of log policy w.r.t theta: outer product of state and (one_hot_action - probs)
        action_one_hot = np.zeros_like(probs)
        action_one_hot[action] = 1.0
        grad_log_pi = np.outer(s, action_one_hot - probs)
        
        # Policy update incorporating the discount factor gamma^t
        theta += alpha_theta * (gamma ** t) * advantage * grad_log_pi
        
        # Baseline update (gradient descent to minimize squared error)
        w += alpha_w * advantage * s
        
    return {
        'theta': theta.tolist(),
        'w': w.tolist(),
        'returns': returns,
        'advantages': advantages
    }