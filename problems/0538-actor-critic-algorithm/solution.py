import numpy as np


def actor_critic_step(
    state_features,
    action,
    reward,
    next_state_features,
    done,
    theta,
    w,
    gamma,
    alpha_theta,
    alpha_w,
):
    """Perform one step of the Actor-Critic algorithm with linear function

    approximation.

    Args:
        state_features: np.ndarray shape (n_features,) - feature vector for
          current state
        action: int - action taken
        reward: float - reward received
        next_state_features: np.ndarray shape (n_features,) - feature vector for
          next state
        done: bool - whether episode terminated
        theta: np.ndarray shape (n_features, n_actions) - actor parameters
        w: np.ndarray shape (n_features,) - critic weight vector
        gamma: float - discount factor
        alpha_theta: float - actor learning rate
        alpha_w: float - critic learning rate

    Returns:
        tuple: (updated_theta, updated_w, td_error, action_probs)
            - updated_theta: np.ndarray shape (n_features, n_actions)
            - updated_w: np.ndarray shape (n_features,)
            - td_error: float - the temporal difference error
            - action_probs: np.ndarray shape (n_actions,) - softmax probabilities
              before update
    """
    state_features = np.array(state_features, dtype=float)
    next_state_features = np.array(next_state_features, dtype=float)
    theta = np.array(theta, dtype=float)
    w = np.array(w, dtype=float)

    # 1. Compute the TD error using the critic's value estimates
    v_s = np.dot(w, state_features)
    v_next_s = 0.0 if done else np.dot(w, next_state_features)
    td_error = float(reward + gamma * v_next_s - v_s)

    # 2. Update the critic weights using semi-gradient TD(0)
    updated_w = w + alpha_w * td_error * state_features

    # 3. Compute softmax action probabilities under the current actor parameters
    logits = np.dot(state_features, theta)  # shape (n_actions,)
    shifted_logits = logits - np.max(logits)
    exp_logits = np.exp(shifted_logits)
    action_probs = exp_logits / np.sum(exp_logits)

    # 4. Update the actor parameters using the policy gradient theorem
    n_actions = theta.shape[1]
    one_hot_action = np.zeros(n_actions)
    one_hot_action[action] = 1.0

    score = np.outer(state_features, one_hot_action - action_probs)
    updated_theta = theta + alpha_theta * td_error * score

    return updated_theta, updated_w, td_error, action_probs