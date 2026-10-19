import numpy as np

def a2c_update(
    num_states: int,
    num_actions: int,
    batch_states: list,
    batch_actions: list,
    batch_rewards: list,
    batch_dones: list,
    bootstrap_states: list,
    gamma: float,
    value_coeff: float,
    entropy_coeff: float,
    alpha_theta: float,
    alpha_w: float,
    theta_init: np.ndarray,
    w_init: np.ndarray
) -> tuple:
    """
    Perform a single A2C batch update from parallel environment data.
    
    Args:
        num_states: Number of states
        num_actions: Number of actions
        batch_states: K lists of T state indices
        batch_actions: K lists of T action indices
        batch_rewards: K lists of T rewards
        batch_dones: K lists of T done flags
        bootstrap_states: K state indices for bootstrapping
        gamma: Discount factor
        value_coeff: Coefficient for value loss
        entropy_coeff: Coefficient for entropy bonus
        alpha_theta: Actor learning rate
        alpha_w: Critic learning rate
        theta_init: Initial actor params, shape (num_states, num_actions)
        w_init: Initial critic weights, shape (num_states,)
    
    Returns:
        Tuple of (w, theta) as numpy arrays
    """
    K = len(batch_states)
    T = len(batch_states[0]) if K > 0 else 0
    N = K * T

    g_theta = np.zeros_like(theta_init, dtype=float)
    g_w = np.zeros_like(w_init, dtype=float)

    def softmax(logits):
        shift_logits = logits - np.max(logits)
        exp_l = np.exp(shift_logits)
        return exp_l / np.sum(exp_l)

    for k in range(K):
        returns = [0.0] * T
        G = w_init[bootstrap_states[k]]
        for t in reversed(range(T)):
            if batch_dones[k][t]:
                G = batch_rewards[k][t]
            else:
                G = batch_rewards[k][t] + gamma * G
            returns[t] = G

        for t in range(T):
            st = batch_states[k][t]
            at = batch_actions[k][t]
            Gt = returns[t]

            vt = w_init[st]
            at_advantage = Gt - vt

            logits = theta_init[st]
            pi = softmax(logits)
            log_pi = np.log(pi + 1e-10)
            H = -np.sum(pi * log_pi)

            one_hot = np.zeros(num_actions, dtype=float)
            one_hot[at] = 1.0

            pg_term = at_advantage * (one_hot - pi)
            entropy_term = -pi * (log_pi + H)

            g_theta[st] += pg_term + entropy_coeff * entropy_term
            g_w[st] += at_advantage

    theta = theta_init + alpha_theta * (g_theta / N)
    w = w_init + alpha_w * value_coeff * (g_w / N)

    return w, theta