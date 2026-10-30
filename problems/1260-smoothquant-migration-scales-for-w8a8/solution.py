import numpy as np

def smoothquant(X: np.ndarray, W: np.ndarray, alpha: float = 0.5):
    """
    Compute SmoothQuant migration scales and smoothed tensors for W8A8 quantization.
    
    Args:
        X: Activations tensor of shape (T, in_features)
        W: Weights tensor of shape (out_features, in_features)
        alpha: Migration strength parameter (default 0.5)
        
    Returns:
        tuple: (s, X_smooth, W_smooth)
    """
    # Compute maximum absolute values across the feature dimension (channels)
    max_X = np.max(np.abs(X), axis=0)  # shape: (in_features,)
    max_W = np.max(np.abs(W), axis=0)  # shape: (in_features,)
    
    # Calculate numerator and denominator with the given formula
    numer = max_X ** alpha
    denom = max_W ** (1 - alpha)
    
    # Apply denominator floor of 1e-8 to prevent division by zero
    denom = np.maximum(denom, 1e-8)
    
    s = numer / denom
    
    # If the numerator is 0, set s_j = 1e-8
    s[numer == 0] = 1e-8
    
    # Apply migration scales: X_smooth = X / s, W_smooth = W * s
    # Broadcasting works natively since s is 1D with size equal to in_features (columns)
    X_smooth = X / s
    W_smooth = W * s
    
    return s, X_smooth, W_smooth