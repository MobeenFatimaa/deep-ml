import numpy as np

def calculate_contrast(img: np.ndarray) -> int:
    """
    Calculate the contrast of a grayscale image using max - min range.
    
    Args:
        img (numpy.ndarray): 2D array representing a grayscale image.
        
    Returns:
        int: Contrast value (max pixel value - min pixel value).
    """
    if img.size == 0:
        return 0
        
    return int(np.max(img) - np.min(img))