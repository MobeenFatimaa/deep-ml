import numpy as np


class LSTM:
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Default weights and biases (can be overwritten by tests)
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)
        
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def forward(
        self,
        x: np.ndarray,
        initial_hidden_state: np.ndarray,
        initial_cell_state: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run a single-layer LSTM over a sequence.

        Args:
            x: Input sequence of shape (T, D)
            initial_hidden_state: Initial hidden state of shape (H, 1)
            initial_cell_state: Initial cell state of shape (H, 1)

        Returns:
            outputs: NumPy array of shape (T, H, 1) holding every h_t
            h_T: Final hidden state of shape (H, 1)
            c_T: Final cell state of shape (H, 1)
        """
        T = x.shape[0]
        H = self.hidden_size
        
        h_prev = initial_hidden_state
        c_prev = initial_cell_state
        
        outputs = np.zeros((T, H, 1))
        
        def sigmoid(z):
            return 1.0 / (1.0 + np.exp(-z))
        
        for t in range(T):
            x_t = x[t].reshape(-1, 1)  # shape (D, 1)
            
            # Stack hidden state first, then input: [h_{t-1}; x_t] of shape (H + D, 1)
            z_t = np.vstack((h_prev, x_t))
            
            f_t = sigmoid(np.dot(self.Wf, z_t) + self.bf)
            i_t = sigmoid(np.dot(self.Wi, z_t) + self.bi)
            c_tilde = np.tanh(np.dot(self.Wc, z_t) + self.bc)
            o_t = sigmoid(np.dot(self.Wo, z_t) + self.bo)
            
            c_t = f_t * c_prev + i_t * c_tilde
            h_t = o_t * np.tanh(c_t)
            
            outputs[t] = h_t
            h_prev = h_t
            c_prev = c_t
            
        return outputs, h_prev, c_prev