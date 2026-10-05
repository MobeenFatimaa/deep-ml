import numpy as np


def rnn_forward(
    input_sequence: list[list[float]],
    initial_hidden_state: list[float],
    Wx: list[list[float]],
    Wh: list[list[float]],
    b: list[float],
) -> list[float]:
    """Processes an input sequence through a simple Vanilla RNN cell and returns the

    final hidden state rounded to 4 decimal places.
    """
    h = np.array(initial_hidden_state, dtype=float)
    Wx_np = np.array(Wx, dtype=float)
    Wh_np = np.array(Wh, dtype=float)
    b_np = np.array(b, dtype=float)

    for x_t in input_sequence:
        x_t_np = np.array(x_t, dtype=float)
        # Recurrent update using column vector convention: h_t = tanh(Wx @ x_t + Wh @ h_{t-1} + b)
        h = np.tanh(np.dot(Wx_np, x_t_np) + np.dot(Wh_np, h) + b_np)

    return np.round(h, 4).tolist()