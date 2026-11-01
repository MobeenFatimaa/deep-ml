import numpy as np

def sparse_window_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, window_size: int, scale_factor: float = None) -> np.ndarray:
    """
    Computes sparse attention over long sequences by sliding a fixed-radius window.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
        window_size: Integer radius w of the attention window
        scale_factor: Optional scaling factor for dot products (defaults to sqrt(d_k))
        
    Returns:
        np.ndarray of shape (seq_len, d_v) containing the attention results.
    """
    seq_len, d_k = Q.shape
    d_v = V.shape[1]
    
    if scale_factor is None:
        scale_factor = np.sqrt(d_k)
        
    output = np.zeros((seq_len, d_v))
    
    for i in range(seq_len):
        # Define the inclusive window bounds [start_idx, end_idx - 1]
        start_idx = max(0, i - window_size)
        end_idx = min(seq_len, i + window_size + 1)
        
        # Slice keys and values for the current window
        K_window = K[start_idx:end_idx]
        V_window = V[start_idx:end_idx]
        
        # Compute scaled dot-product attention scores within the window
        scores = np.dot(K_window, Q[i]) / scale_factor
        
        # Apply stable softmax
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)
        weights = exp_scores / np.sum(exp_scores)
        
        # Compute the weighted sum of values in the window
        output[i] = np.dot(weights, V_window)
        
    return output