import numpy as np

def off_policy_nstep_sarsa(
    states: list,
    actions: list,
    rewards: list,
    Q: dict,
    target_policy: dict,
    behavior_policy: dict,
    n: int,
    gamma: float,
    alpha: float
) -> dict:
    """
    Perform off-policy n-step Sarsa with importance sampling on a single episode.
    
    Args:
        states: List of states visited in the episode (length T+1)
        actions: List of actions taken (length T)
        rewards: List of rewards received (length T)
        Q: Action-value function as dict {(state, action): value}
        target_policy: Target policy {state: {action: prob}}
        behavior_policy: Behavior policy {state: {action: prob}}
        n: Number of steps for the return
        gamma: Discount factor
        alpha: Learning rate
    
    Returns:
        Updated Q dictionary
    """
    T = len(actions)
    
    for t in range(T):
        h = min(t + n, T)
        
        # Compute n-step return G
        G = 0.0
        for i in range(t, h):
            G += (gamma ** (i - t)) * rewards[i]
            
        if h < T:
            sh = states[h]
            ah = actions[h]
            G += (gamma ** (h - t)) * Q.get((sh, ah), 0.0)
            
        # Compute importance sampling ratio rho
        rho = 1.0
        for i in range(t + 1, min(t + n, T)):
            st = states[i]
            at = actions[i]
            pi_prob = target_policy.get(st, {}).get(at, 0.0)
            b_prob = behavior_policy.get(st, {}).get(at, 1.0)
            if b_prob == 0.0:
                rho = 0.0
                break
            rho *= pi_prob / b_prob
            
        # Update Q(S_t, A_t)
        st = states[t]
        at = actions[t]
        old_q = Q.get((st, at), 0.0)
        Q[(st, at)] = old_q + alpha * rho * (G - old_q)
        
    return Q