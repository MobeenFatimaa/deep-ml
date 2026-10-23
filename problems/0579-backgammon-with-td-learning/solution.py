import numpy as np

def td_gammon_learning(game_states, outcome, W1, b1, w2, b2, alpha, lambd):
    """
    Apply TD(lambda) learning to update a neural network value function
    over a single self-play game sequence.
    
    Args:
        game_states: list of numpy arrays, board features at each time step
        outcome: float, final game result (1.0 = win, 0.0 = loss)
        W1: np.ndarray of shape (n_hidden, n_features), hidden weights
        b1: np.ndarray of shape (n_hidden,), hidden biases
        w2: np.ndarray of shape (n_hidden,), output weights
        b2: float, output bias
        alpha: float, learning rate
        lambd: float, eligibility trace decay (lambda)
    
    Returns:
        Tuple of (W1, b1, w2, b2) after all TD updates
    """
    if not game_states:
        return W1, b1, w2, b2

    def sigmoid(x):
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))

    def forward_and_grad(x, W1, b1, w2, b2):
        z1 = W1 @ x + b1
        h = sigmoid(z1)
        z2 = np.dot(w2, h) + b2
        V = sigmoid(z2)
        
        dv_dz2 = V * (1.0 - V)
        dv_dw2 = dv_dz2 * h
        dv_db2 = dv_dz2
        
        dh_dz1 = h * (1.0 - h)
        delta_h = dv_dz2 * w2 * dh_dz1
        
        dv_dW1 = np.outer(delta_h, x)
        dv_db1 = delta_h
        
        return V, dv_dW1, dv_db1, dv_dw2, dv_db2

    e_W1 = np.zeros_like(W1)
    e_b1 = np.zeros_like(b1)
    e_w2 = np.zeros_like(w2)
    e_b2 = 0.0

    T = len(game_states)

    for t in range(T):
        x_t = game_states[t]
        V_t, dv_dW1, dv_db1, dv_dw2, dv_db2 = forward_and_grad(x_t, W1, b1, w2, b2)

        if t < T - 1:
            x_next = game_states[t + 1]
            V_next, _, _, _, _ = forward_and_grad(x_next, W1, b1, w2, b2)
            delta = V_next - V_t
        else:
            delta = outcome - V_t

        e_W1 = lambd * e_W1 + dv_dW1
        e_b1 = lambd * e_b1 + dv_db1
        e_w2 = lambd * e_w2 + dv_dw2
        e_b2 = lambd * e_b2 + dv_db2

        W1 = W1 + alpha * delta * e_W1
        b1 = b1 + alpha * delta * e_b1
        w2 = w2 + alpha * delta * e_w2
        b2 = b2 + alpha * delta * e_b2

    return W1, b1, w2, b2