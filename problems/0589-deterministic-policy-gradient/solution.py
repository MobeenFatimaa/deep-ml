import numpy as np

def deterministic_policy_gradient(states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray, theta: np.ndarray, w: np.ndarray, gamma: float, alpha_theta: float, alpha_w: float) -> tuple:
    """
    Perform one update step of the Deterministic Policy Gradient algorithm
    with linear function approximation for both the actor and critic.
    
    Args:
        states: Batch of states, shape (N, state_dim)
        actions: Batch of actions taken, shape (N,)
        rewards: Batch of rewards, shape (N,)
        next_states: Batch of next states, shape (N, state_dim)
        dones: Batch of terminal flags, shape (N,)
        theta: Policy parameters, shape (state_dim,)
        w: Critic parameters, shape (state_dim + 1,)
        gamma: Discount factor
        alpha_theta: Policy learning rate
        alpha_w: Critic learning rate
    
    Returns:
        Tuple of (theta_new, w_new) as lists rounded to 4 decimal places
    """
    N = states.shape[0]
    
    # 1. Critic Update
    # Predict next actions using current policy
    next_actions = next_states @ theta
    
    # Compute Q(s', a') using critic weights
    q_next = next_states @ w[:-1] + next_actions * w[-1]
    
    # Compute TD targets
    targets = rewards + gamma * (1.0 - dones) * q_next
    
    # Compute Q(s, a) for the actual taken actions
    q_current = states @ w[:-1] + actions * w[-1]
    
    # Compute TD errors
    td_errors = targets - q_current
    
    # Construct feature matrix phi(s, a)
    phi = np.hstack([states, actions[:, None]])
    
    # Compute mean semi-gradient for critic
    critic_grad = np.mean(td_errors[:, None] * phi, axis=0)
    w_new = w + alpha_w * critic_grad
    
    # 2. Actor Update
    # Using pre-update critic weights, dQ/da = w[-1] and nabla_theta mu(s) = states
    policy_grads = w[-1] * states
    mean_policy_grad = np.mean(policy_grads, axis=0)
    
    theta_new = theta + alpha_theta * mean_policy_grad
    
    return (
        np.round(theta_new, 4).tolist(),
        np.round(w_new, 4).tolist()
    )