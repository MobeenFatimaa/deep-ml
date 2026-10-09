import numpy as np

def flash_attention_forward(Q: np.ndarray, K: np.ndarray, V: np.ndarray, 
                            block_size: int = 2) -> np.ndarray:
    """
    Compute attention output using Flash Attention v1 algorithm.
    
    Args:
        Q: Query matrix (N, d) where N is seq_len, d is d_model
        K: Key matrix (N, d)
        V: Value matrix (N, d)
        block_size: Size of blocks for tiled computation (B_r and B_c)
    
    Returns:
        Output matrix (N, d) matching standard scaled dot-product attention
    """
    N, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    
    # Block sizes (B_r for query rows, B_c for key/value blocks)
    B_r = block_size
    B_c = block_size
    
    # Number of query blocks (Tr) and key/value blocks (Tc)
    T_r = int(np.ceil(N / B_r))
    T_c = int(np.ceil(N / B_c))
    
    # Initialize output O, running max m, and running row sum l
    O = np.zeros((N, d), dtype=np.float64)
    m = np.full((N, 1), -np.inf, dtype=np.float64)
    l = np.zeros((N, 1), dtype=np.float64)
    
    # Outer loop over key/value blocks
    for j in range(T_c):
        K_j = K[j * B_c : (j + 1) * B_c]  # Shape: (B_c, d)
        V_j = V[j * B_c : (j + 1) * B_c]  # Shape: (B_c, d)
        
        # Inner loop over query blocks
        for i in range(T_r):
            q_start, q_end = i * B_r, min((i + 1) * B_r, N)
            
            Q_i = Q[q_start:q_end]            # Shape: (B_r, d)
            O_i = O[q_start:q_end]            # Shape: (B_r, d)
            m_i = m[q_start:q_end]            # Shape: (B_r, 1)
            l_i = l[q_start:q_end]            # Shape: (B_r, 1)
            
            # 1. Compute block attention scores S_ij = (Q_i @ K_j^T) * scale
            S_ij = (Q_i @ K_j.T) * scale      # Shape: (B_r, B_c)
            
            # 2. Compute block row-wise maximum m_ij_tilde
            m_ij_tilde = np.max(S_ij, axis=1, keepdims=True)  # Shape: (B_r, 1)
            
            # 3. Compute unnormalized exponentials P_ij_tilde
            P_ij_tilde = np.exp(S_ij - m_ij_tilde)           # Shape: (B_r, B_c)
            
            # 4. Compute block row sums l_ij_tilde
            l_ij_tilde = np.sum(P_ij_tilde, axis=1, keepdims=True)  # Shape: (B_r, 1)
            
            # 5. Compute new running maximum m_i_new
            m_i_new = np.maximum(m_i, m_ij_tilde)                   # Shape: (B_r, 1)
            
            # 6. Rescale factors for updating accumulated statistics
            alpha = np.exp(m_i - m_i_new)
            beta = np.exp(m_ij_tilde - m_i_new)
            
            # 7. Update running sum l_i_new
            l_i_new = alpha * l_i + beta * l_ij_tilde
            
            # 8. Update output block O_i
            # O_i_new = (l_i * exp(m_i - m_i_new) * O_i + exp(m_ij_tilde - m_i_new) * (P_ij_tilde @ V_j)) / l_i_new
            O_i_new = (alpha * l_i * O_i + beta * (P_ij_tilde @ V_j)) / l_i_new
            
            # Save updated statistics back to global arrays
            O[q_start:q_end] = O_i_new
            m[q_start:q_end] = m_i_new
            l[q_start:q_end] = l_i_new

    return O