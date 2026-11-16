import numpy as np


def compute_block_partial(q, K_block, V_block):
    """Compute online-softmax partial (m_b, l_b, a_b) for a block of KV rows."""
    d = q.shape[0]
    scale = 1.0 / np.sqrt(d)
    scores = np.dot(K_block, q) * scale
    m_b = np.max(scores)
    exp_s = np.exp(scores - m_b)
    l_b = np.sum(exp_s)
    a_b = np.sum(exp_s[:, None] * V_block, axis=0)
    return m_b, l_b, a_b


def merge_partials(m0, l0, a0, m1, l1, a1):
    """Merge two online-softmax partials."""
    if m0 == -np.inf:
        return m1, l1, a1
    if m1 == -np.inf:
        return m0, l0, a0
        
    m = max(m0, m1)
    w0 = np.exp(m0 - m)
    w1 = np.exp(m1 - m)
    l = l0 * w0 + l1 * w1
    a = a0 * w0 + a1 * w1
    return m, l, a


def attention_single_pass(q, K, V, block):
    """
    Whole-sequence decode attention on one 'SM': walk the KV cache in blocks of `block` rows,
    form each block's online-softmax partial and merge it into the running state in order.
    Returns acc / sum with shape (d,).
    """
    n, d = K.shape
    m = -np.inf
    l = 0.0
    a = np.zeros(d, dtype=K.dtype)
    
    for i in range(0, n, block):
        K_b = K[i : i + block]
        V_b = V[i : i + block]
        m_b, l_b, a_b = compute_block_partial(q, K_b, V_b)
        m, l, a = merge_partials(m, l, a, m_b, l_b, a_b)
        
    return (a / l).astype(K.dtype)


def attention_split_kv(q, K, V, block, num_splits):
    """
    Split-KV decode attention: blocks are grouped and block partials are merged in
    global block order, guaranteeing bitwise equality with attention_single_pass.
    """
    n, d = K.shape
    num_blocks = (n + block - 1) // block
    if num_blocks == 0:
        return np.zeros(d, dtype=K.dtype)
        
    blocks_per_split = int(np.ceil(num_blocks / num_splits))
    
    # Compute block partials across all blocks
    block_partials = []
    for b_idx in range(num_blocks):
        start = b_idx * block
        end = min(start + block, n)
        K_b = K[start:end]
        V_b = V[start:end]
        block_partials.append(compute_block_partial(q, K_b, V_b))
        
    # Merge all block partials in global block order
    m = -np.inf
    l = 0.0
    a = np.zeros(d, dtype=K.dtype)
    
    for m_b, l_b, a_b in block_partials:
        m, l, a = merge_partials(m, l, a, m_b, l_b, a_b)
        
    return (a / l).astype(K.dtype)


def batch_invariant_attention(Q, K, V, block, num_sms):
    """
    Q: (B, d), K, V: (B, n, d). Uses single-pass if B >= num_sms, else split-KV.
    Returns (out, kernel) where out has shape (B, d) in K's dtype and kernel in {"single", "split"}.
    """
    B, d = Q.shape
    n = K.shape[1]
    num_blocks = (n + block - 1) // block
    
    out = np.zeros((B, d), dtype=K.dtype)
    
    if B >= num_sms:
        kernel = "single"
        for i in range(B):
            out[i] = attention_single_pass(Q[i], K[i], V[i], block)
    else:
        kernel = "split"
        num_splits = max(1, min(num_blocks, num_sms // B))
        for i in range(B):
            out[i] = attention_split_kv(Q[i], K[i], V[i], block, num_splits)
            
    return out, kernel