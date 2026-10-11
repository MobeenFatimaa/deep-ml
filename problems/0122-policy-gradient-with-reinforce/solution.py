import numpy as np


def compute_policy_gradient(
    theta: np.ndarray, episodes: list[list[tuple[int, int, float]]]
) -> np.ndarray:
    """Estimate the policy gradient using REINFORCE.

    Args:
        theta: (num_states x num_actions) policy parameters.
        episodes: List of episodes, where each episode is a list of (state, action, reward).

    Returns:
        Average policy gradient (same shape as theta).
    """
    grad_sum = np.zeros_like(theta)

    for episode in episodes:
        T = len(episode)
        returns = np.zeros(T)
        G = 0.0

        # Compute discounted returns from back to front (with gamma = 1.0)
        for t in reversed(range(T)):
            G = G + episode[t][2]
            returns[t] = G

        for t in range(T):
            s, a, _ = episode[t]
            G_t = returns[t]

            # Softmax policy over theta[s, :]
            logits = theta[s, :]
            shifted = logits - np.max(logits)
            exp_logits = np.exp(shifted)
            probs = exp_logits / np.sum(exp_logits)

            # Gradient of log pi(a|s) w.r.t theta[s, :]
            grad_log_pi = -probs
            grad_log_pi[a] += 1.0

            grad_sum[s, :] += G_t * grad_log_pi

    # Average over the number of episodes
    return grad_sum / len(episodes)