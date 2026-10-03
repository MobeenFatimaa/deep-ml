import numpy as np

def to_categorical(x, n_col=None):
    if n_col is None:
        n_col = np.max(x) + 1
        
    # Option 1: Using np.eye for concise vectorization
    return np.eye(n_col, dtype=float)[x]