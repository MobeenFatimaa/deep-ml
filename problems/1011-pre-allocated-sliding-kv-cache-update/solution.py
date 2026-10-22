import numpy as np

def sliding_kv_cache_update(cache_k, cache_v, current_len, new_k, new_v, window_size):
    """
    Update a pre-allocated sliding-window KV cache and produce the causal mask
    for the new queries against the updated cache.

    Args:
        cache_k, cache_v: nested lists / arrays of shape
            (batch, num_heads, window_size, head_dim).
        current_len: int, number of valid entries currently in the cache (<= window_size).
        new_k, new_v: nested lists / arrays of shape
            (batch, num_heads, new_len, head_dim).
        window_size: int, maximum cache length.

    Returns:
        dict with keys 'cache_k', 'cache_v', 'current_len', 'mask'.
    """
    cache_k_out = np.array(cache_k, dtype=float).copy()
    cache_v_out = np.array(cache_v, dtype=float).copy()
    new_k = np.array(new_k, dtype=float)
    new_v = np.array(new_v, dtype=float)
    
    new_len = new_k.shape[2]
    updated_len = min(current_len + new_len, window_size)
    
    # Extract valid entries from current cache and concatenate with new entries
    old_valid_k = cache_k_out[:, :, :current_len, :]
    old_valid_v = cache_v_out[:, :, :current_len, :]
    
    concat_k = np.concatenate([old_valid_k, new_k], axis=2)
    concat_v = np.concatenate([old_valid_v, new_v], axis=2)
    
    # Keep the most recent window_size entries (or updated_len entries)
    surviving_k = concat_k[:, :, -updated_len:, :]
    surviving_v = concat_v[:, :, -updated_len:, :]
    
    # Write into the output cache
    cache_k_out[:, :, :updated_len, :] = surviving_k
    cache_v_out[:, :, :updated_len, :] = surviving_v
    
    # Build causal mask for new queries against the updated cache
    mask = np.zeros((new_len, updated_len), dtype=int)
    for i in range(new_len):
        pos_i = max(updated_len - new_len + i, 0)
        for j in range(updated_len):
            if j <= pos_i:
                mask[i, j] = 1
                
    return {
        'cache_k': cache_k_out.tolist(),
        'cache_v': cache_v_out.tolist(),
        'current_len': int(updated_len),
        'mask': mask.tolist()
    }