import numpy as np

def deterministic_sparse_attention_backward(contributions: list, num_sms: int, kv_len: int, dim: int) -> list:
    """
    Deterministically accumulate sparse attention KV gradient contributions.

    Args:
        contributions: list of dicts with keys 'sm_id' (int), 'kv_idx' (int),
                       'grad' (list[float] of length dim)
        num_sms: number of SMs (size of per-SM buffer axis)
        kv_len: KV sequence length
        dim: feature dimension

    Returns:
        Nested list of shape (kv_len, dim) containing the reduced gradient.
    """
    # 1. Maintain a separate accumulation buffer per SM: shape (num_sms, kv_len, dim)
    sm_buffers = np.zeros((num_sms, kv_len, dim), dtype=np.float64)
    
    # 2. Within each SM's buffer, accumulate contributions in the order they appear
    for contrib in contributions:
        sm_id = contrib['sm_id']
        kv_idx = contrib['kv_idx']
        grad = contrib['grad']
        
        sm_buffers[sm_id, kv_idx] += grad
        
    # 3. Reduce the per-SM buffers into the final output by iterating SM ids in ascending order
    final_grad = np.sum(sm_buffers, axis=0)  # sums along the num_sms axis in order (0, 1, 2, ...)
    
    return final_grad.tolist()