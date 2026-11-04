import numpy as np

def global_avg_pool(x: np.ndarray) -> np.ndarray:
    """
    Perform Global Average Pooling on a 3D NumPy array of shape (height, width, channels).
    
    Args:
        x (numpy.ndarray): 3D feature map array.
        
    Returns:
        numpy.ndarray: 1D array of shape (channels,) containing channel averages.
    """
    x = np.asarray(x)
    
    # Compute the mean across the spatial dimensions (height and width)
    return np.mean(x, axis=(0, 1))