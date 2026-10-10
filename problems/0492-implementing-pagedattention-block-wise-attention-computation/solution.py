import numpy as np

def paged_attention(
    query: np.ndarray,
    key_cache: np.ndarray,
    value_cache: np.ndarray,
    block_table: list,
    context_len: int
) -> np.ndarray:
    """
    Perform scaled dot-product attention with paged KV cache.

    Args:
        query:      (num_heads, head_dim) query for a single token
        key_cache:   (num_physical_blocks, block_size, num_heads, head_dim)
        value_cache: (num_physical_blocks, block_size, num_heads, head_dim)
        block_table: list of physical block indices (logical -> physical)
        context_len: number of valid KV tokens in the sequence

    Returns:
        (num_heads, head_dim) attention output, rounded to 4 decimal places
    """
    num_heads, head_dim = query.shape
    num_physical_blocks, block_size, _, _ = key_cache.shape

    # Gather keys and values up to context_len using the block_table
    gathered_keys_list = []
    gathered_values_list = []

    tokens_collected = 0
    for logical_block_idx in block_table:
        if tokens_collected >= context_len:
            break
        
        physical_block_idx = logical_block_idx
        # Determine how many tokens to take from this block
        tokens_in_this_block = min(block_size, context_len - tokens_collected)
        
        # Extract valid slots from the physical block
        k_block = key_cache[physical_block_idx, :tokens_in_this_block]  # (tokens_in_this_block, num_heads, head_dim)
        v_block = value_cache[physical_block_idx, :tokens_in_this_block]  # (tokens_in_this_block, num_heads, head_dim)
        
        gathered_keys_list.append(k_block)
        gathered_values_list.append(v_block)
        
        tokens_collected += tokens_in_this_block

    # Concatenate along the sequence dimension (axis 0)
    # Shape becomes (context_len, num_heads, head_dim)
    keys = np.concatenate(gathered_keys_list, axis=0)
    values = np.concatenate(gathered_values_list, axis=0)

    # Transpose to group by head: (num_heads, context_len, head_dim)
    keys = np.transpose(keys, (1, 0, 2))
    values = np.transpose(values, (1, 0, 2))

    # Expand query for broadcasting/multi-head computation: (num_heads, 1, head_dim)
    q = np.expand_dims(query, axis=1)

    # Compute scaled dot-product attention scores per head: (num_heads, 1, context_len)
    scale = 1.0 / np.sqrt(head_dim)
    scores = np.matmul(q, np.transpose(keys, (0, 2, 1))) * scale

    # Numerically stable softmax along the context dimension
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Compute weighted sum of values: (num_heads, 1, head_dim)
    output = np.matmul(attn_weights, values)

    # Squeeze out the sequence dimension to match (num_heads, head_dim) and round
    output = np.squeeze(output, axis=1)
    return np.round(output, 4)