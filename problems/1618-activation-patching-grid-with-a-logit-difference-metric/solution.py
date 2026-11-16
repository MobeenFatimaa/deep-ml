import numpy as np


def reader_metric(R, W_r, d_vec):
    """
    Attention from the last position over all positions, then d_vec . output.
    """
    # R shape: (P, d)
    # Query vector from the last position r_{P-1}
    r_last = R[-1]
    
    # Compute attention scores: s_p = r_{P-1}^T W_r r_p
    # r_last @ W_r has shape (d,), dotted with R.T gives shape (P,)
    scores = np.dot(r_last, W_r) @ R.T
    
    # Softmax over positions with numerical stability
    max_score = np.max(scores)
    exp_scores = np.exp(scores - max_score)
    attn_weights = exp_scores / np.sum(exp_scores)
    
    # Compute attended mixture output o = sum_p a_p r_p
    output = np.sum(attn_weights[:, None] * R, axis=0)
    
    # Compute final metric m = d_vec . o
    m = float(np.dot(d_vec, output))
    return m


def assemble(embed, writes):
    """
    Assemble residuals from embeddings and component writes: embed + writes.sum(axis=0).
    """
    return embed + np.sum(writes, axis=0)


def patching_grid(embed_clean, writes_clean, embed_corrupt, writes_corrupt, W_r, d_vec, decimals=4):
    """
    Compute (C, P) normalised restoration scores via activation patching.
    """
    m_clean = reader_metric(assemble(embed_clean, writes_clean), W_r, d_vec)
    m_corrupt = reader_metric(assemble(embed_corrupt, writes_corrupt), W_r, d_vec)
    
    denom = m_clean - m_corrupt
    C, P, d = writes_clean.shape
    grid = np.zeros((C, P), dtype=float)
    
    for c in range(C):
        for p in range(P):
            # Copy corrupt writes and patch entry [c, p] with clean write
            writes_patched = writes_corrupt.copy()
            writes_patched[c, p] = writes_clean[c, p]
            
            # Assemble patched run using corrupt embedding
            R_patched = assemble(embed_corrupt, writes_patched)
            m_patched = reader_metric(R_patched, W_r, d_vec)
            
            if denom == 0:
                score = 0.0
            else:
                score = (m_patched - m_corrupt) / denom
                
            grid[c, p] = round(float(score), decimals)
            
    return grid


def best_patch(grid):
    """
    Find the (c, p) of the largest grid entry (first occurrence in row-major order) and its value.
    """
    flat_idx = np.argmax(grid)
    c, p = np.unravel_index(flat_idx, grid.shape)
    value = float(grid[c, p])
    return (int(c), int(p), value)