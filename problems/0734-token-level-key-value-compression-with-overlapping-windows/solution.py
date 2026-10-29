import numpy as np

def compressed_kv_entries(H, W_aKV, W_bKV, W_aZ, W_bZ, B_a, B_b, m, stride):
    """
    H: (n, d) hidden states
    W_aKV, W_bKV: (d, c) KV projection weights for the two streams
    W_aZ, W_bZ: (d, c) compression-weight projection matrices
    B_a, B_b: (m, c) learnable positional biases
    m: window size (tokens per group per stream)
    stride: step between consecutive windows

    Returns: list of lists representing the compressed KV entries
             of shape (num_groups, c) where num_groups = (n - m) // stride + 1
    """
    H = np.array(H, dtype=float)
    W_aKV = np.array(W_aKV, dtype=float)
    W_bKV = np.array(W_bKV, dtype=float)
    W_aZ = np.array(W_aZ, dtype=float)
    W_bZ = np.array(W_bZ, dtype=float)
    B_a = np.array(B_a, dtype=float)
    B_b = np.array(B_b, dtype=float)
    
    n, d = H.shape
    num_groups = (n - m) // stride + 1
    compressed_results = []
    
    for i in range(num_groups):
        start = i * stride
        H_window = H[start : start + m] # (m, d)
        
        # Stream a
        KV_a = H_window @ W_aKV # (m, c)
        Z_a = H_window @ W_aZ + B_a # (m, c)
        
        # Stream b
        KV_b = H_window @ W_bKV # (m, c)
        Z_b = H_window @ W_bZ + B_b # (m, c)
        
        # Combine across both streams (2m elements)
        KV_all = np.concatenate([KV_a, KV_b], axis=0) # (2m, c)
        Z_all = np.concatenate([Z_a, Z_b], axis=0) # (2m, c)
        
        # Numerically stable softmax across the 2m elements per feature dimension
        max_z = np.max(Z_all, axis=0, keepdims=True)
        exp_z = np.exp(Z_all - max_z)
        weights = exp_z / np.sum(exp_z, axis=0, keepdims=True) # (2m, c)
        
        # Weighted sum of KV entries
        compressed = np.sum(weights * KV_all, axis=0) # (c,)
        compressed_results.append(compressed.tolist())
        
    return compressed_results