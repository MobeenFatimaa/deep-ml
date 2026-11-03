import numpy as np

def flash_decoding(q, k, v, n_splits):
    """
    FlashDecoding attention for one decode step with GQA.

    q: (batch, n_q_heads, head_dim)
    k, v: (batch, seq, n_kv_heads, head_dim)
    n_splits: int
    Returns: (batch, n_q_heads, head_dim)
    """
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    batch, n_q_heads, head_dim = q.shape
    _, seq, n_kv_heads, _ = k.shape
    
    g = n_q_heads // n_kv_heads
    scale = 1.0 / np.sqrt(head_dim)
    
    # Partition seq indices using numpy.array_split
    splits = np.array_split(range(seq), n_splits)
    
    output = np.zeros((batch, n_q_heads, head_dim), dtype=np.float64)
    
    for b in range(batch):
        for h in range(n_q_heads):
            kv_h = h // g
            qh = q[b, h] # shape: (head_dim,)
            
            split_ms = []
            split_ls = []
            split_os = []
            
            # Compute partial attention results across each sequence split
            for split_indices in splits:
                if len(split_indices) == 0:
                    split_ms.append(-np.inf)
                    split_ls.append(0.0)
                    split_os.append(np.zeros(head_dim, dtype=np.float64))
                    continue
                
                k_split = k[b, split_indices, kv_h, :]  # shape: (len_split, head_dim)
                v_split = v[b, split_indices, kv_h, :]  # shape: (len_split, head_dim)
                
                # Scaled dot-product scores: (len_split,)
                scores = np.matmul(k_split, qh) * scale
                
                m_s = np.max(scores)
                exp_scores = np.exp(scores - m_s)
                l_s = np.sum(exp_scores)
                
                # Weighted sum of values for this split: (head_dim,)
                o_s = np.dot(exp_scores, v_split)
                
                split_ms.append(m_s)
                split_ls.append(l_s)
                split_os.append(o_s)
            
            # Merge partial results using online softmax reduction
            max_m = max(split_ms) if split_ms else -0.0
            if max_m == -np.inf or max_m is None:
                continue
                
            total_l = 0.0
            total_o = np.zeros(head_dim, dtype=np.float64)
            
            for m_s, l_s, o_s in zip(split_ms, split_ls, split_os):
                if l_s == 0.0:
                    continue
                factor = np.exp(m_s - max_m)
                total_l += l_s * factor
                total_o += o_s * factor
                
            if total_l > 0:
                output[b, h] = total_o / total_l
            else:
                output[b, h] = 0.0
                
    return output