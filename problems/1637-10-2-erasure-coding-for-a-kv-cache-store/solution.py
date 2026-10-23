import numpy as np

P = 257


def encode_stripe(data, n=12):
    """Evaluate the data polynomials at x = 1..n modulo 257; return an (n, m) array."""
    data = np.array(data, dtype=np.int64)
    k, m = data.shape
    
    j_vals = np.arange(1, n + 1, dtype=np.int64)
    i_vals = np.arange(k, dtype=np.int64)
    
    V = np.power(j_vals[:, None], i_vals[None, :], dtype=np.int64) % P
    out = (V @ data) % P
    return out.astype(np.int64)


def decode_stripe(chunks, k=10):
    """Recover the (k, m) data array from any k surviving chunks, or None."""
    surviving_indices = []
    surviving_chunks = []
    for idx, chunk in enumerate(chunks):
        if chunk is not None:
            surviving_indices.append(idx)
            surviving_chunks.append(chunk)
            
    if len(surviving_chunks) < k:
        return None
        
    selected_indices = surviving_indices[:k]
    selected_chunks = np.array(surviving_chunks[:k], dtype=np.int64)
    
    x_vals = np.array([idx + 1 for idx in selected_indices], dtype=np.int64)
    i_vals = np.arange(k, dtype=np.int64)
    V_sub = np.power(x_vals[:, None], i_vals[None, :], dtype=np.int64) % P
    
    m = selected_chunks.shape[1]
    # Augmented matrix [V_sub | selected_chunks] of shape (k, k + m)
    M_aug = np.hstack([V_sub, selected_chunks]).astype(np.int64) % P
    
    for col in range(k):
        pivot_row = -1
        for row in range(col, k):
            if M_aug[row, col] != 0:
                pivot_row = row
                break
        if pivot_row == -1:
            return None
            
        if pivot_row != col:
            M_aug[[col, pivot_row]] = M_aug[[pivot_row, col]]
            
        pivot_val = M_aug[col, col]
        inv_pivot = pow(int(pivot_val), P - 2, P)
        M_aug[col] = (M_aug[col] * inv_pivot) % P
        
        for row in range(k):
            if row != col:
                factor = M_aug[row, col]
                if factor != 0:
                    M_aug[row] = (M_aug[row] - factor * M_aug[col]) % P
                    
    data_res = M_aug[:, k:] % P
    return data_res.astype(np.int64)


def usable_fraction(n, k):
    """Share of raw capacity holding data."""
    return round(float(k) / float(n), 4)