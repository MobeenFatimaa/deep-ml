import numpy as np

def transfer_and_attend(X, W_q_lin, W_k_lin, W_v_lin):
    """
    Transfer linear-style weights (shape (d_out, d_in)) into parameter-style
    weights (shape (d_in, d_out)) and compute single-head causal self-attention.

    Args:
        X: input array of shape (seq_len, d_in)
        W_q_lin, W_k_lin, W_v_lin: linear-style weight matrices of shape (d_out, d_in)

    Returns:
        Nested list of shape (seq_len, d_out): the self-attention output.
    """
    X = np.array(X)
    W_q_lin = np.array(W_q_lin)
    W_k_lin = np.array(W_k_lin)
    W_v_lin = np.array(W_v_lin)

    # Transpose weights from (d_out, d_in) to (d_in, d_out)
    W_q = W_q_lin.T
    W_k = W_k_lin.T
    W_v = W_v_lin.T
    
    # Compute projections
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    
    d_out = W_q.shape[1]
    
    # Scaled dot-product scores
    S = (Q @ K.T) / np.sqrt(d_out)
    
    # Apply causal mask (key index > query index set to -10^9)
    seq_len = X.shape[0]
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
    S = S + mask
    
    # Numerically stable softmax
    max_S = np.max(S, axis=-1, keepdims=True)
    exp_S = np.exp(S - max_S)
    softmax_S = exp_S / np.sum(exp_S, axis=-1, keepdims=True)
    
    # Compute attention output
    output = softmax_S @ V
    
    return output.tolist()