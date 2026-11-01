import numpy as np

def group_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, num_groups: int, epsilon: float = 1e-5) -> np.ndarray:
    """
    Perform Group Normalization on a 4D input tensor.
    
    Args:
        X: Input tensor of shape (B, C, H, W)
        gamma: Scale parameter of shape (C,)
        beta: Shift parameter of shape (C,)
        num_groups: Number of groups to divide channels into
        epsilon: Small value for numerical stability
        
    Returns:
        np.ndarray: Normalized and scaled tensor of shape (B, C, H, W)
    """
    B, C, H, W = X.shape
    C_per_group = C // num_groups
    
    # 1. Reshape X to isolate groups: (B, num_groups, C_per_group, H, W)
    X_reshaped = X.reshape(B, num_groups, C_per_group, H, W)
    
    # 2. Compute mean and variance along each group's channels, height, and width axes
    mean = np.mean(X_reshaped, axis=(2, 3, 4), keepdims=True)
    var = np.var(X_reshaped, axis=(2, 3, 4), keepdims=True)
    
    # 3. Normalize within each group
    X_norm = (X_reshaped - mean) / np.sqrt(var + epsilon)
    
    # 4. Reshape back to original 4D shape: (B, C, H, W)
    X_norm = X_norm.reshape(B, C, H, W)
    
    # 5. Apply learnable scale (gamma) and shift (beta), broadcasted across (B, H, W)
    gamma_reshaped = gamma.reshape(1, C, 1, 1)
    beta_reshaped = beta.reshape(1, C, 1, 1)
    
    return X_norm * gamma_reshaped + beta_reshaped