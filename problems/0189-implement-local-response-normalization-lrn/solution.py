import numpy as np

def local_response_normalization(x: np.ndarray, n: int = 5, k: float = 2.0, alpha: float = 1e-4, beta: float = 0.75) -> np.ndarray:
    """
    Applies Local Response Normalization across the channel dimension.

    Args:
        x: Input tensor of shape (N, C, H, W)
        n: Local window size
        k: Additive constant
        alpha: Scaling parameter
        beta: Exponent parameter

    Returns:
        Normalized tensor of same shape as input, rounded to 4 decimal places.
    """
    h = n // 2
    N, C, H, W = x.shape
    
    # Square the input values
    x_sq = x ** 2
    
    # Pad along the channel axis (axis 1) with h zeros on both sides
    pad_width = [(0, 0), (h, h), (0, 0), (0, 0)]
    padded = np.pad(x_sq, pad_width, mode='constant', constant_values=0.0)
    
    # Compute the sum of squares in the local window for each channel using correct clipped bounds
    sum_sq = np.zeros_like(x)
    for i in range(C):
        start = max(0, i - h) + h
        end = min(C - 1, i + h) + h + 1
        sum_sq[:, i, :, :] = np.sum(padded[:, start:end, :, :], axis=1)
        
    # Apply the LRN formula
    denominator = (k + alpha * sum_sq) ** beta
    out = x / denominator
    
    return np.round(out, 4)