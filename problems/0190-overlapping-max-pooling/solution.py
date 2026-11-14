import numpy as np
import math

def overlapping_max_pool2d(x: np.ndarray, kernel_size: int = 3, stride: int = 2) -> np.ndarray:
    """
    Applies overlapping max pooling to a 4D tensor (N, C, H, W).
    Uses ceil mode for output dimensions (allows partial windows at boundaries).

    Args:
        x: Input array of shape (N, C, H, W)
        kernel_size: Size of pooling window (int)
        stride: Stride between pooling windows (int), must be < kernel_size

    Returns:
        A 4D tensor after overlapping pooling with ceil mode.
    """
    N, C, H, W = x.shape
    
    # Compute output dimensions using ceil mode
    H_out = math.ceil((H - kernel_size) / stride) + 1
    W_out = math.ceil((W - kernel_size) / stride) + 1
    
    # Initialize output array
    out = np.empty((N, C, H_out, W_out), dtype=x.dtype)
    
    # Slide the pooling window across the spatial dimensions
    for i in range(H_out):
        h_start = i * stride
        h_end = min(h_start + kernel_size, H)
        
        for j in range(W_out):
            w_start = j * stride
            w_end = min(w_start + kernel_size, W)
            
            # Extract the window region and compute max over spatial axes (H and W)
            window = x[:, :, h_start:h_end, w_start:w_end]
            out[:, :, i, j] = np.max(window, axis=(2, 3))
            
    C = out
    return out