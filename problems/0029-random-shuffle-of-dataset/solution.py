import numpy as np

def shuffle_data(X, y, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # Generate a random permutation of sample indices
    permutation = np.random.permutation(X.shape[0])
    
    # Apply the same index permutation to both arrays
    return X[permutation], y[permutation]