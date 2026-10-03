import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True):
    n_samples = X.shape[0]
    
    # Determine subset size based on replacement flag
    if replacements:
        subset_size = n_samples
    else:
        subset_size = n_samples // 2
        
    subsets = []
    
    for _ in range(n_subsets):
        # Generate random indices for sampling
        indices = np.random.choice(n_samples, size=subset_size, replace=replacements)
        
        # Sample X and y, converting result to Python lists
        X_subset = X[indices].tolist()
        y_subset = y[indices].tolist()
        
        subsets.append((X_subset, y_subset))
        
    return subsets