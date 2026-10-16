import numpy as np

def guidance_attention_mask(
    chunk_sizes: list[int],
    current_chunk: int
) -> np.ndarray:
    """
    Build a boolean attention mask for chunked autoregressive video generation.

    Args:
        chunk_sizes:   tokens per chunk, in order
        current_chunk: index of the chunk being generated

    Returns:
        Boolean array of shape (total_tokens, total_tokens).
        True means the row token can attend to the column token.
    """
    total_tokens = sum(chunk_sizes)
    mask = np.zeros((total_tokens, total_tokens), dtype=bool)
    
    # Compute chunk boundaries
    starts = [0] + list(np.cumsum(chunk_sizes)[:-1])
    ends = list(np.cumsum(chunk_sizes))
    
    for i, (start_i, end_i) in enumerate(zip(starts, ends)):
        for j, (start_j, end_j) in enumerate(zip(starts, ends)):
            # Historical chunks attending to historical chunks (bidirectional)
            if i < current_chunk and j < current_chunk:
                mask[start_i:end_i, start_j:end_j] = True
            
            # Current chunk attending to historical chunks (full access)
            elif i == current_chunk and j < current_chunk:
                mask[start_i:end_i, start_j:end_j] = True
            
            # Current chunk attending to itself (causal)
            elif i == current_chunk and j == current_chunk:
                chunk_len = end_i - start_i
                mask[start_i:end_i, start_j:end_j] = np.tril(np.ones((chunk_len, chunk_len), dtype=bool))
                
    return mask