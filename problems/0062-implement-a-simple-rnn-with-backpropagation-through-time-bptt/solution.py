import numpy as np


class SimpleRNN:

    def __init__(self, input_size, hidden_size, output_size):
        """Initializes the RNN with random weights and zero biases."""
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01

        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

    def forward(self, x):
        """Forward pass through the RNN for a given sequence of inputs."""
        self.x_seq = []
        self.h_seq = {-1: np.zeros((self.hidden_size, 1))}
        self.y_seq = []
        outputs = []

        for t, x_t in enumerate(x):
            x_t = np.array(x_t, dtype=float).reshape(-1, 1)
            self.x_seq.append(x_t)

            # Hidden state update: h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
            h_t = np.tanh(
                np.dot(self.W_xh, x_t)
                + np.dot(self.W_hh, self.h_seq[t - 1])
                + self.b_h
            )
            self.h_seq[t] = h_t

            # Output prediction: y_t = W_hy @ h_t + b_y
            y_t = np.dot(self.W_hy, h_t) + self.b_y
            self.y_seq.append(y_t)
            outputs.append(y_t)

        return np.array(outputs)

    def backward(self, x, y, learning_rate):
        """Backpropagation through time (BPTT) to adjust weights based on error gradient."""
        T = len(self.x_seq)

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        dh_next = np.zeros((self.hidden_size, 1))

        # Backpropagate through time
        for t in reversed(range(T)):
            y_t_target = np.array(y[t], dtype=float).reshape(-1, 1)

            # Derivative of 1/2 * MSE Loss w.r.t linear output y_t
            dy_t = self.y_seq[t] - y_t_target

            # Output layer gradients
            dW_hy += np.dot(dy_t, self.h_seq[t].T)
            db_y += dy_t

            # Accumulated gradient for hidden state h_t
            dh_t = np.dot(self.W_hy.T, dy_t) + dh_next

            # Derivative through tanh activation function
            dh_raw = (1 - self.h_seq[t] ** 2) * dh_t

            # Hidden state parameter gradients
            db_h += dh_raw
            dW_xh += np.dot(dh_raw, self.x_seq[t].T)
            dW_hh += np.dot(dh_raw, self.h_seq[t - 1].T)

            # Recurrent gradient passed to step t-1
            dh_next = np.dot(self.W_hh.T, dh_raw)

        # SGD parameter update
        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y