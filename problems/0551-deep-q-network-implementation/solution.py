import numpy as np


def dqn_training_step(
    weights: dict,
    target_weights: dict,
    batch: dict,
    gamma: float,
    learning_rate: float,
) -> tuple:
    """Perform one DQN training step with backpropagation.

    Network: input -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> Q-values

    Args:
        weights: dict with 'W1', 'b1', 'W2', 'b2'
        target_weights: same structure as weights, for the target network
        batch: dict with 'states', 'actions', 'rewards', 'next_states', 'dones'
        gamma: discount factor
        learning_rate: learning rate for SGD

    Returns:
        tuple of (updated_weights: dict, loss: float rounded to 4 decimals)
    """
    states = np.array(batch["states"], dtype=float)
    actions = np.array(batch["actions"], dtype=int)
    rewards = np.array(batch["rewards"], dtype=float)
    next_states = np.array(batch["next_states"], dtype=float)
    dones = np.array(batch["dones"], dtype=bool)

    W1, b1 = weights["W1"], weights["b1"]
    W2, b2 = weights["W2"], weights["b2"]

    # 1. Forward pass for current states
    z1 = states @ W1 + b1
    h1 = np.maximum(0, z1)  # ReLU
    q_values = h1 @ W2 + b2  # (batch_size, num_actions)

    batch_size = states.shape[0]
    # 2. Extract Q-values corresponding to actions taken
    predicted_q = q_values[np.arange(batch_size), actions]

    # 3. Compute target Q-values using target network
    t_W1, t_b1 = target_weights["W1"], target_weights["b1"]
    t_W2, t_b2 = target_weights["W2"], target_weights["b2"]

    next_z1 = next_states @ t_W1 + t_b1
    next_h1 = np.maximum(0, next_z1)
    next_q_values = next_h1 @ t_W2 + t_b2
    max_next_q = np.max(next_q_values, axis=1)

    targets = rewards + gamma * max_next_q * (~dones)

    # 4. Compute MSE loss
    errors = predicted_q - targets
    loss = float(np.mean(errors**2))

    # 5. Backpropagation
    # Derivative of loss wrt predicted_q: dL/d(predicted_q) = 2 * (predicted_q - targets) / batch_size
    d_q = 2.0 * errors / batch_size  # (batch_size,)

    # Gradient wrt W2 and b2
    # q_values = h1 @ W2 + b2 => dL/dW2 = h1.T @ d_q (for selected actions)
    d_q_full = np.zeros_like(q_values)
    d_q_full[np.arange(batch_size), actions] = d_q

    dW2 = h1.T @ d_q_full
    db2 = np.sum(d_q_full, axis=0)

    # Gradient wrt h1: dL/dh1 = d_q_full @ W2.T
    dh1 = d_q_full @ W2.T

    # Backprop through ReLU
    dz1 = dh1 * (z1 > 0)

    # Gradient wrt W1 and b1
    dW1 = states.T @ dz1
    db1 = np.sum(dz1, axis=0)

    # 6. Update weights using gradient descent
    updated_weights = {
        "W1": W1 - learning_rate * dW1,
        "b1": b1 - learning_rate * db1,
        "W2": W2 - learning_rate * dW2,
        "b2": b2 - learning_rate * db2,
    }

    return updated_weights, round(loss, 4)