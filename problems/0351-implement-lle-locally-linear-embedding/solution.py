import numpy as np

def lle(X: np.ndarray, n_neighbors: int, n_components: int) -> np.ndarray:
    """
    Perform Locally Linear Embedding for dimensionality reduction.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        n_neighbors: Number of nearest neighbors to use
        n_components: Target dimensionality
    
    Returns:
        Embedding Y of shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    
    # Step 1: Find k nearest neighbors using Euclidean distance
    dists = np.sum(X**2, axis=1, keepdims=True) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
    neighbors = np.argsort(dists, axis=1)[:, 1:n_neighbors + 1]
    
    # Step 2: Compute reconstruction weights W
    W = np.zeros((n_samples, n_samples))
    reg = 1e-3
    
    for i in range(n_samples):
        nbrs = neighbors[i]
        Z = X[nbrs] - X[i]
        C = np.dot(Z, Z.T)
        
        # Fixed regularization added directly to the diagonal
        C += reg * np.eye(n_neighbors)
        
        # Solve for weights summing to 1
        w = np.linalg.solve(C, np.ones(n_neighbors))
        w = w / np.sum(w)
        W[i, nbrs] = w
        
    # Step 3: Compute low-dimensional embedding Y
    I = np.eye(n_samples)
    M = np.dot((I - W).T, (I - W))
    
    evals, evecs = np.linalg.eigh(M)
    sort_indices = np.argsort(evals)
    evecs = evecs[:, sort_indices]
    
    # Discard the smallest eigenvector and take the next n_components
    Y = evecs[:, 1:n_components + 1]
    
    # Step 4: Standardize column signs so the sum of each column is positive
    for j in range(n_components):
        if np.sum(Y[:, j]) < 0:
            Y[:, j] *= -1
            
    return Y