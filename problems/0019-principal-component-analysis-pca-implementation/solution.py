import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    # Step 1: Standardize the dataset (mean = 0, std = 1)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    
    # Step 2: Compute covariance matrix
    # rowvar=False indicates that columns represent features
    cov_matrix = np.cov(standardized_data, rowvar=False)
    
    # Step 3: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Step 4: Sort eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Select the top k principal components
    top_k_components = sorted_eigenvectors[:, :k]
    
    # Step 5: Apply Sign Convention
    for i in range(k):
        col = top_k_components[:, i]
        # Find first element with absolute value > 1e-10
        nonzero_mask = np.abs(col) > 1e-10
        if np.any(nonzero_mask):
            first_nonzero_idx = np.where(nonzero_mask)[0][0]
            if col[first_nonzero_idx] < 0:
                top_k_components[:, i] *= -1
                
    return np.round(top_k_components, 4)