import numpy as np

def td_position_eval(
    states: list,
    outcome: float,
    W1: list,
    b1: list,
    W2: list,
    b2: list,
    alpha: float,
    gamma: float,
    lam: float
) -> dict:
    """
    Train a position evaluation network using online TD(lambda).

    Args:
        states: List of feature vectors, each a list of floats.
        outcome: Terminal game outcome (float).
        W1: Hidden weights, shape (n_hidden, n_features).
        b1: Hidden biases, shape (n_hidden,).
        W2: Output weights, shape (1, n_hidden).
        b2: Output bias, shape (1,).
        alpha: Learning rate.
        gamma: Discount factor.
        lam: Eligibility trace decay parameter.

    Returns:
        Dict with 'W1', 'b1', 'W2', 'b2' as lists, rounded to 4 decimals.
    """
    W1 = np.array(W1, dtype=float)
    b1 = np.array(b1, dtype=float)
    W2 = np.array(W2, dtype=float)
    b2 = np.array(b2, dtype=float)

    e_W1 = np.zeros_like(W1)
    e_b1 = np.zeros_like(b1)
    e_W2 = np.zeros_like(W2)
    e_b2 = np.zeros_like(b2)

    T = len(states)

    def forward(x):
        z1 = W1 @ x + b1
        h = 1.0 / (1.0 + np.exp(-z1))
        z2 = W2 @ h + b2
        v = 1.0 / (1.0 + np.exp(-z2[0]))
        return z1, h, z2, v

    for t in range(T):
        x_t = np.array(states[t], dtype=float)
        z1, h, z2, v_t = forward(x_t)

        dv_dz2 = v_t * (1.0 - v_t)
        dv_dW2 = dv_dz2 * h.reshape(1, -1)
        dv_db2 = np.array([dv_dz2])

        dv_dh = W2.T * dv_dz2
        dh_dz1 = h * (1.0 - h)
        dv_dz1 = dv_dh.ravel() * dh_dz1

        dv_dW1 = np.outer(dv_dz1, x_t)
        dv_db1 = dv_dz1

        e_W1 = gamma * lam * e_W1 + dv_dW1
        e_b1 = gamma * lam * e_b1 + dv_db1
        e_W2 = gamma * lam * e_W2 + dv_dW2
        e_b2 = gamma * lam * e_b2 + dv_db2

        if t < T - 1:
            x_next = np.array(states[t + 1], dtype=float)
            _, _, _, v_next = forward(x_next)
            delta_t = gamma * v_next - v_t
        else:
            delta_t = outcome - v_t

        W1 += alpha * delta_t * e_W1
        b1 += alpha * delta_t * e_b1
        W2 += alpha * delta_t * e_W2
        b2 += alpha * delta_t * e_b2

    return {
        'W1': np.round(W1, 4).tolist(),
        'b1': np.round(b1, 4).tolist(),
        'W2': np.round(W2, 4).tolist(),
        'b2': np.round(b2, 4).tolist(),
    }