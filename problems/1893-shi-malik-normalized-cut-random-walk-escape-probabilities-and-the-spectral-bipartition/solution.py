import numpy as np

def ncut_value(W, labels):
    """Computes Shi and Malik's normalized cut sum_k W(S_k, S_bar_k) / vol(S_k)."""
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    d = np.sum(W, axis=1)
    
    total_ncut = 0.0
    for k in unique_labels:
        S = (labels == k)
        S_bar = ~S
        
        # W(S, S_bar) is the sum of weights between S and S_bar
        cut_k = np.sum(W[np.ix_(S, S_bar)])
        vol_k = np.sum(d[S])
        
        if vol_k > 0:
            total_ncut += cut_k / vol_k
            
    return float(total_ncut)

def escape_probabilities(W, in_S):
    """Returns (p(S_bar | S), p(S | S_bar)) for the random walk starting from the stationary distribution."""
    in_S = np.array(in_S, dtype=bool)
    in_S_bar = ~in_S
    
    d = np.sum(W, axis=1)
    
    vol_S = np.sum(d[in_S])
    vol_S_bar = np.sum(d[in_S_bar])
    
    cut_S_Sbar = np.sum(W[np.ix_(in_S, in_S_bar)])
    
    p_out = float(cut_S_Sbar / vol_S) if vol_S > 0 else 0.0
    p_in = float(cut_S_Sbar / vol_S_bar) if vol_S_bar > 0 else 0.0
    
    return (p_out, p_in)

def shi_malik_bipartition(W):
    """Solves the Shi-Malik spectral bipartition and finds the optimal threshold."""
    N = W.shape[0]
    d = np.sum(W, axis=1)
    
    # Avoid division by zero if any degree is 0
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    
    L = np.diag(d) - W
    M = D_inv_sqrt @ L @ D_inv_sqrt
    
    # Compute eigenvalues and eigenvectors of symmetric matrix M using np.linalg.eigh
    eigenvals, eigenvecs = np.linalg.eigh(M)
    
    # Second-smallest eigenvalue corresponds to index 1 (since index 0 is 0 eigenvalue)
    v = eigenvecs[:, 1]
    u = D_inv_sqrt @ v
    
    # Flip sign so that its entry of largest absolute value is positive
    max_abs_idx = np.argmax(np.abs(u))
    if u[max_abs_idx] < 0:
        u = -u
        
    # Search the splitting point: distinct values of u except the largest
    sorted_u = np.sort(np.unique(u))
    # Exclude the largest value
    thresholds = sorted_u[:-1]
    
    best_ncut = float('inf')
    best_labels = np.zeros(N, dtype=int)
    
    for t in thresholds:
        labels = (u > t).astype(int)  # 0 for u_i <= t, 1 for u_i > t
        if np.all(labels == 0) or np.all(labels == 1):
            continue
        
        current_ncut = ncut_value(W, labels)
        if current_ncut < best_ncut:
            best_ncut = current_ncut
            best_labels = labels
            
    # Ensure node 0 is in class 0; if not, flip the labels
    if best_labels[0] == 1:
        best_labels = 1 - best_labels
        
    return (best_labels.tolist(), float(best_ncut), u)