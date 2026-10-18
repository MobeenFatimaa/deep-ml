import numpy as np


def reinforce_with_baseline(episode, theta, w, gamma, alpha_theta, alpha_w):
    """Perform one episode update of REINFORCE with a value function baseline.

    Args:
        episode: list of (state, action, reward) tuples
        theta: np.ndarray of shape (n_states, n_actions), policy parameters
        w: np.ndarray of shape (n_states,), value function parameters
        gamma: float, discount factor
        alpha_theta: float, policy learning rate
        alpha_w: float, value function learning rate

    Returns:
        tuple: (theta_new, w_new) updated parameters
    """
    theta = np.array(theta, dtype=float)
    w = np.array(w, dtype=float)
    T = len(episode)

    # 1. Compute discounted returns G_t for each timestep
    returns = np.zeros(T, dtype=float)
    G = 0.0
    for t in range(T - 1, -1, -1):
        _, _, reward = episode[t]
        G = reward + gamma * G
        returns[t] = G

    # 2. Process each timestep sequentially
    for t in range(T):
        state, action, _ = episode[t]
        G_t = returns[t]

        # Current value estimate
        v_s = w[state]

        # Advantage estimate
        delta = G_t - v_s

        # Update value parameters
        w[state] += alpha_w * delta

        # Compute softmax action probabilities from theta for current state
        logits = theta[state, :]
        shifted_logits = logits - np.max(logits)
        exp_logits = np.exp(shifted_logits)
        pi = exp_logits / np.sum(exp_logits)

        # Policy gradient (score function) for the taken action
        grad_ln_pi = np.zeros_like(pi)
        grad_ln_pi[action] = 1.0
        grad_ln_pi -= pi

        # Update policy parameters scaled by discount factor gamma^t and advantage
        discount = gamma**t
        theta[state, :] += alpha_theta * discount * delta * grad_ln_pi

    return theta, w