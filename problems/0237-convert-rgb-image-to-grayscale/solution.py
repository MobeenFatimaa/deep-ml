import numpy as np

def rgb_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminosity method.
    
    Args:
        image: RGB image as list or numpy array of shape (H, W, 3)
               with values in range [0, 255]
    
    Returns:
        Grayscale image as 2D list with integer values,
        or -1 if input is invalid
    """
    # 1. Convert to numpy array safely to check shape and types
    try:
        arr = np.asarray(image)
    except Exception:
        return -1

    # 2. Check dimensions and shape (H, W, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return -1

    H, W, C = arr.shape
    if H == 0 or W == 0:
        return -1

    # 3. Check pixel value range [0, 255]
    if np.any(arr < 0) or np.any(arr > 255):
        return -1

    # 4. Apply luminosity formula: 0.299*R + 0.587*G + 0.114*B
    R = arr[:, :, 0]
    G = arr[:, :, 1]
    B = arr[:, :, 2]

    gray = 0.299 * R + 0.587 * G + 0.114 * B

    # 5. Round to nearest integer and convert to 2D Python list
    gray_rounded = np.round(gray).astype(int)
    return gray_rounded.tolist()