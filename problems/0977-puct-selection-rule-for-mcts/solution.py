import numpy as np

def puct_select(Q, N_children, P, N_parent, c_puct):
    """Return the index of the child maximizing the PUCT score."""
    Q = np.asarray(Q, dtype=np.float64)
    N_children = np.asarray(N_children, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    
    # Calculate exploration term: c_puct * P * (sqrt(N_parent) / (1 + N_children))
    exploration = c_puct * P * (np.sqrt(N_parent) / (1.0 + N_children))
    
    # Calculate PUCT score for each child
    scores = Q + exploration
    
    # Return the index of the maximum score (np.argmax returns the smallest index on ties)
    return int(np.argmax(scores))