import numpy as np

def lookback_compressed_kv(H, W_aKV, W_bKV, W_aZ, W_bZ, B_a, B_b, m):
    """
    H: (n, d) hidden states; n must be divisible by m
    W_aKV, W_bKV: (d, c) KV projection weights (current, look-back)
    W_aZ, W_bZ: (d, c) compression-weight projection matrices
    B_a, B_b: (m, c) learnable positional biases
    m: block size

    Returns: list of lists of shape (n // m, c) with the compressed KV entries.
    """
    H = np.array(H, dtype=float)
    W_aKV = np.array(W_aKV, dtype=float)
    W_bKV = np.array(W_bKV, dtype=float)
    W_aZ = np.array(W_aZ, dtype=float)
    W_bZ = np.array(W_bZ, dtype=float)
    B_a = np.array(B_a, dtype=float)
    B_b = np.array(B_b, dtype=float)
    
    n, d = H.shape
    num_blocks = n // m
    
    H_blocks = H.reshape(num_blocks, m, d)
    
    KV_a = np.matmul(H_blocks, W_aKV)
    logits_a = np.matmul(H_blocks, W_aZ) + B_a[np.newaxis, :, :]
    
    KV_b = np.matmul(H_blocks, W_bKV)
    logits_b = np.matmul(H_blocks, W_bZ) + B_b[np.newaxis, :, :]
    
    result = []
    for i in range(num_blocks):
        kv_a_i = KV_a[i]
        logits_a_i = logits_a[i]
        
        if i == 0:
            kv_b_i = np.zeros_like(kv_a_i)
            logits_b_i = np.full_like(logits_a_i, -np.inf)
        else:
            kv_b_i = KV_b[i - 1]
            logits_b_i = logits_b[i - 1]
            
        kv_combined = np.concatenate([kv_a_i, kv_b_i], axis=0)
        logits_combined = np.concatenate([logits_a_i, logits_b_i], axis=0)
        
        max_logits = np.max(logits_combined, axis=0, keepdims=True)
        exp_logits = np.exp(logits_combined - max_logits)
        weights = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
        
        compressed_col = np.sum(weights * kv_combined, axis=0)
        result.append(compressed_col.tolist())
        
    return result