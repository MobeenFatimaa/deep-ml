import numpy as np

def smote(X_minority: np.ndarray, n_synthetic: int, k: int = 5) -> np.ndarray:
    """
    Generate synthetic samples using SMOTE algorithm.

    Note: the random seed is set by the grader before your function runs,
    so you do NOT need to set it. Just use numpy's global RNG directly
    (np.random.randint, np.random.random, ...).

    Args:
        X_minority: 2D array of minority class samples (n_samples, n_features)
        n_synthetic: Number of synthetic samples to generate
        k: Number of nearest neighbors to consider

    Returns:
        2D array of synthetic samples (n_synthetic, n_features)
    """
    X_minority = np.asarray(X_minority, dtype=float)
    n_samples, n_features = X_minority.shape
    
    k_actual = min(k, n_samples - 1)
    
    # Return empty array of shape (0, n_features) if conditions are met
    if k_actual == 0 or n_synthetic == 0:
        return np.empty((0, n_features), dtype=float)
    
    synthetic_samples = []
    
    for _ in range(n_synthetic):
        # 1. Draw base sample
        i = np.random.randint(0, n_samples)
        x_i = X_minority[i]
        
        # 2. Compute Euclidean distances to all minority samples
        distances = np.linalg.norm(X_minority - x_i, axis=1)
        
        # Exclude x_i itself by setting distance to infinity or slicing after sorting
        distances[i] = np.inf
        
        # Get indices of the k_actual nearest neighbors (ordered by ascending distance)
        nearest_indices = np.argsort(distances)[:k_actual]
        
        # 3. Select neighbor index from [0, k_actual)
        j = np.random.randint(0, k_actual)
        x_nn = X_minority[nearest_indices[j]]
        
        # 4. Draw gap and interpolate
        gap = np.random.random()
        x_synthetic = x_i + gap * (x_nn - x_i)
        
        synthetic_samples.append(x_synthetic)
        
    return np.array(synthetic_samples)