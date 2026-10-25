import numpy as np

def off_policy_td_lambda(
    episodes: list,
    behavior_policy: list,
    target_policy: list,
    num_states: int,
    num_actions: int,
    gamma: float,
    lam: float,
    alpha: float
) -> np.ndarray:
    """
    Off-policy TD(lambda) prediction with importance-weighted eligibility traces.

    Args:
        episodes: List of episodes, each a list of (state, action, reward) tuples.
        behavior_policy: b(a|s) as 2D list of shape (num_states, num_actions).
        target_policy: pi(a|s) as 2D list of shape (num_states, num_actions).
        num_states: Number of states.
        num_actions: Number of actions.
        gamma: Discount factor.
        lam: Lambda parameter for trace decay.
        alpha: Learning rate.

    Returns:
        V: numpy array of shape (num_states,) with estimated state values.
    """
    V = np.zeros(num_states, dtype=float)

    for episode in episodes:
        e = np.zeros(num_states, dtype=float)
        T = len(episode)
        for t in range(T):
            s, a, r = episode[t]
            
            # Importance sampling ratio
            rho = target_policy[s][a] / behavior_policy[s][a]
            
            # Next state value
            if t + 1 < T:
                next_s = episode[t + 1][0]
                v_next = V[next_s]
            else:
                v_next = 0.0
            
            # TD error
            delta = r + gamma * v_next - V[s]
            
            # Update eligibility traces: decay, add indicator, scale by rho
            e = gamma * lam * e
            e[s] += 1.0
            e = rho * e
            
            # Update value estimates
            V += alpha * delta * e

    return V