import numpy as np

def n_step_tree_backup(Q, states, actions, rewards, target_policy, terminal_states, n, alpha, gamma):
    """
    Implement the n-step tree backup algorithm for a single episode.
    
    Args:
        Q: np.ndarray of shape (num_states, num_actions), initial Q-values
        states: list of int, states visited (length T+1)
        actions: list of int, actions taken (length T)
        rewards: list of float, rewards received (length T)
        target_policy: np.ndarray of shape (num_states, num_actions), target policy pi(a|s)
        terminal_states: list of int, terminal state indices
        n: int, number of backup steps
        alpha: float, learning rate
        gamma: float, discount factor
    
    Returns:
        Q: np.ndarray of shape (num_states, num_actions), updated Q-values rounded to 4 decimals
    """
    Q = np.array(Q, dtype=float)
    target_policy = np.array(target_policy, dtype=float)
    T = len(actions)
    
    for t in range(T):
        tau = min(t + n, T)
        if states[tau] in terminal_states:
            G = 0.0
        else:
            s_tau = states[tau]
            G = np.sum(target_policy[s_tau] * Q[s_tau])
            
        for k in range(tau - 1, t, -1):
            sk = states[k]
            ak = actions[k]
            rk = rewards[k]
            
            policy_sum = np.sum(target_policy[sk] * Q[sk])
            sum_not_ak = policy_sum - target_policy[sk, ak] * Q[sk, ak]
            
            G = sum_not_ak + target_policy[sk, ak] * (rk + gamma * G)
            
        st = states[t]
        at = actions[t]
        rt = rewards[t]
        
        G = rt + gamma * G
        
        Q[st, at] += alpha * (G - Q[st, at])
        
    return np.round(Q, 4)