def attention_memory(T: int, H: int, W: int, num_heads: int, bytes_per_element: int) -> list:
    """
    Compute memory (in bytes) of attention score tensors for full space-time
    attention and for space-time factorized attention.

    Args:
        T: number of temporal frames
        H: spatial height in tokens
        W: spatial width in tokens
        num_heads: number of attention heads
        bytes_per_element: bytes per stored score value (e.g. 4 for fp32, 2 for fp16)

    Returns:
        [full_bytes, factorized_bytes]
    """
    S = H * W
    N = T * S
    
    # Full attention: N x N score matrix per head
    full_bytes = num_heads * (N * N) * bytes_per_element
    
    # Factorized attention: 
    # - T spatial matrices of size S x S
    # - S temporal matrices of size T x T
    factorized_bytes = num_heads * (T * S * S + S * T * T) * bytes_per_element
    
    return [full_bytes, factorized_bytes]