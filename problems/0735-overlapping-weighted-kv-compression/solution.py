import numpy as np

def overlapping_kv_compression(Z_a, Z_b, B_a, B_b, C_a, C_b):
    """
    Compute compressed KV entries from two overlapping score/value blocks.

    Returns:
        Nested list of shape (N, D).
    """
    Z_a = np.array(Z_a, dtype=float)
    Z_b = np.array(Z_b, dtype=float)
    B_a = np.array(B_a, dtype=float)
    B_b = np.array(B_b, dtype=float)
    C_a = np.array(C_a, dtype=float)
    C_b = np.array(C_b, dtype=float)
    
    # 1. Add positional bias vectors B_a and B_b to Z_a and Z_b respectively
    biased_Z_a = Z_a + B_a  # shape (N, M_a)
    biased_Z_b = Z_b + B_b  # shape (N, M_b)
    
    # 2. Concatenate the biased scores into a single vector of length M_a + M_b
    Z_concat = np.concatenate([biased_Z_a, biased_Z_b], axis=1)  # shape (N, M_a + M_b)
    
    # 3. Apply a numerically-stable softmax over the concatenated vector across the last dimension
    max_z = np.max(Z_concat, axis=-1, keepdims=True)
    exp_z = np.exp(Z_concat - max_z)
    weights = exp_z / np.sum(exp_z, axis=-1, keepdims=True)  # shape (N, M_a + M_b)
    
    # 4. Split the weights back into W_a (first M_a entries) and W_b (remaining M_b entries)
    M_a = Z_a.shape[1]
    W_a = weights[:, :M_a]  # shape (N, M_a)
    W_b = weights[:, M_a:]  # shape (N, M_b)
    
    # 5. Produce the compressed output C_Comp[i] = W_a @ C_a + W_b @ C_b
    C_comp = (W_a @ C_a) + (W_b @ C_b)
    
    return C_comp.tolist()