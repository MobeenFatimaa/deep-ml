import math

def kv_block_layout(num_tokens: int, m_csa: int, m_hca: int, min_block_size: int) -> list:
    """
    Compute a block layout for a heterogeneous compressed KV cache.

    Returns:
        [block_size, num_blocks, k1, k2, padded_tokens]
    """
    # Compute Least Common Multiple of m_csa and m_hca
    lcm = math.lcm(m_csa, m_hca)
    
    # Target minimum block size must be at least lcm
    target_min = max(min_block_size, lcm)
    
    # Choose smallest multiple of lcm that is >= target_min
    B = math.ceil(target_min / lcm) * lcm
    
    # Number of blocks needed using ceiling division
    num_blocks = math.ceil(num_tokens / B) if num_tokens > 0 else 0
    
    # Number of compressed tokens per block for each stream
    k1 = B // m_csa
    k2 = B // m_hca
    
    # Total raw token capacity after block alignment
    padded_tokens = num_blocks * B
    
    return [B, num_blocks, k1, k2, padded_tokens]