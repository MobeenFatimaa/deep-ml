import numpy as np

def zero_pad_image(img, pad_width):
    """
    Add zero padding around a grayscale image.
    
    Args:
        img: 2D list or numpy array of pixel values
        pad_width: integer number of pixels to pad on each side
    
    Returns:
        Padded image as 2D list with integer values,
        or -1 if input is invalid
    """
    # 1. Validate pad_width
    if not isinstance(pad_width, int) or isinstance(pad_width, bool) or pad_width < 0:
        return -1

    # 2. Convert img to numpy array safely
    try:
        arr = np.asarray(img)
    except Exception:
        return -1

    # 3. Check for valid 2D array
    if arr.ndim != 2:
        return -1

    # 4. Check for empty dimensions
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        return -1

    # 5. Apply zero padding to top, bottom, left, and right
    padded_arr = np.pad(arr, pad_width=pad_width, mode='constant', constant_values=0)

    # 6. Return as a 2D list of integers
    return padded_arr.astype(int).tolist()