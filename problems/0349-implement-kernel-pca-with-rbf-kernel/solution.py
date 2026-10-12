import numpy as np

def kernel_pca_rbf(X: np.ndarray, n_components: int, gamma: float) -> np.ndarray:
    """
    Perform Kernel PCA with RBF kernel.
    
    Args:
        X: Input data matrix of shape (n_samples, n_features)
        n_components: Number of principal components to return
        gamma: RBF kernel parameter
    
    Returns:
        Transformed data of shape (n_samples, n_components)
    """
    n_samples = X.shape[0]
    
    # 1. Compute the RBF kernel matrix
    sq_norms = np.sum(X**2, axis=1, keepdims=True)
    K = np.exp(-gamma * (sq_norms + sq_norms.T - 2 * np.dot(X, X.T)))
    
    # 2. Center the kernel matrix
    one_n = np.ones((n_samples, n_samples)) / n_samples
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    
    # 3. Perform eigendecomposition of K_centered
    eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top n_components and filter out non-positive eigenvalues
    eigenvalues = eigenvalues[:n_components]
    eigenvectors = eigenvectors[:, :n_components]
    
    eigenvalues = np.clip(eigenvalues, 0, None)
    
    # 4. Enforce sign convention: element with largest absolute value in each eigenvector column is positive
    for i in range(n_components):
        v = eigenvectors[:, i]
        max_abs_idx = np.argmax(np.abs(v))
        if v[max_abs_idx] < 0:
            eigenvectors[:, i] = -v
            
    # 5. Project data
    # Standard Kernel PCA projection: X_kpca = alpha * sqrt(lambda) where alpha = eigenvectors / sqrt(lambda) = eigenvectors * 1/sqrt(lambda)
    # Thus X_kpca = eigenvectors * sqrt(lambda)
    # To handle potential sign conventions consistent with standard libraries, ensure eigenvectors are scaled by sqrt(eigenvalues).
    X_transformed = eigenvectors * np.sqrt(eigenvalues)
    
    return X_transformed