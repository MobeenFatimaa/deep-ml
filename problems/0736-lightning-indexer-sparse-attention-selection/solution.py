import numpy as np

def lightning_indexer_topk(H_q, K_comp, W_down, W_up, w_head, k):
    """
    Implement the lightning indexer that selects the top-k most relevant
    compressed key-value entries for each query token.

    Args:
        H_q: list of shape (n_q, d) - query hidden states
        K_comp: list of shape (n_kv, c) - compressed indexer keys
        W_down: list of shape (d, r)
        W_up: list of shape (r, h*c)
        w_head: list of length h
        k: integer number of top entries to select

    Returns:
        nested Python list of shape (n_q, min(k, n_kv)) of integer indices
    """
    H_q = np.array(H_q, dtype=float)
    K_comp = np.array(K_comp, dtype=float)
    W_down = np.array(W_down, dtype=float)
    W_up = np.array(W_up, dtype=float)
    w_head = np.array(w_head, dtype=float)
    
    n_q, d = H_q.shape
    n_kv, c = K_comp.shape
    h = len(w_head)
    
    # 1. Low-rank projection and reshape into heads
    Z = np.matmul(H_q, W_down)          # (n_q, r)
    U = np.matmul(Z, W_up)              # (n_q, h * c)
    queries = U.reshape(n_q, h, c)      # (n_q, h, c)
    
    # 2. Compute index scores and apply ReLU
    # queries: (n_q, h, c), K_comp: (n_kv, c) -> dot product: (n_q, h, n_kv)
    scores = np.matmul(queries, K_comp.T)
    scores = np.maximum(0.0, scores)
    
    # 3. Combine per-head scores using w_head weights
    # scores: (n_q, h, n_kv), w_head: (h,) -> combined: (n_q, n_kv)
    combined_scores = np.sum(scores * w_head[None, :, None], axis=1)
    
    # 4. Top-k selection with tie-breaking (prefer smaller index)
    effective_k = min(k, n_kv)
    result = []
    for row in combined_scores:
        sorted_indices = sorted(range(n_kv), key=lambda idx: (-row[idx], idx))
        result.append(sorted_indices[:effective_k])
        
    return result