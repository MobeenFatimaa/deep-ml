import numpy as np


def actor_critic_traces(
    num_states: int,
    num_actions: int,
    episode: list,
    gamma: float,
    lambda_w: float,
    lambda_theta: float,
    alpha_w: float,
    alpha_theta: float,
    w_init: np.ndarray,
    theta_init: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Episodic actor-critic with eligibility traces.

    Args:
        num_states: Number of states
        num_actions: Number of actions
        episode: List of (state, action, reward, next_state, done) tuples
        gamma: Discount factor
        lambda_w: Trace decay for critic
        lambda_theta: Trace decay for actor
        alpha_w: Critic learning rate
        alpha_theta: Actor learning rate
        w_init: Initial critic weights, shape (num_states,)
        theta_init: Initial actor parameters, shape (num_states, num_actions)

    Returns:
        Tuple of (w, theta) as numpy arrays
    """
    w = np.array(w_init, dtype=float)
    theta = np.array(theta_init, dtype=float)

    z_w = np.zeros(num_states, dtype=float)
    z_theta = np.zeros((num_states, num_actions), dtype=float)
    I = 1.0

    for state, action, reward, next_state, done in episode:
        # 1. Compute value estimates
        v_s = w[state]
        v_next_s = 0.0 if done else w[next_state]

        # 2. Compute TD error
        delta = reward + gamma * v_next_s - v_s

        # 3. Update critic eligibility trace and weights
        z_w = gamma * lambda_w * z_w
        z_w[state] += 1.0
        w += alpha_w * delta * z_w

        # 4. Compute softmax policy probabilities for the current state
        logits = theta[state, :]
        shifted_logits = logits - np.max(logits)
        exp_logits = np.exp(shifted_logits)
        pi = exp_logits / np.sum(exp_logits)

        # 5. Compute policy gradient (score function) for the taken action
        grad_ln_pi = np.zeros(num_actions)
        grad_ln_pi[action] = 1.0
        grad_ln_pi -= pi

        # 6. Update actor eligibility trace and parameters
        z_theta = gamma * lambda_theta * z_theta
        z_theta[state, :] += I * grad_ln_pi
        theta += alpha_theta * delta * z_theta

        # 7. Update discount accumulator
        I *= gamma

    return w, theta