import numpy as np

def bilinear_resize(image, new_height: int, new_width: int) -> list:
    """
    Resize a grayscale (2D) or RGB (3D) image using bilinear interpolation.
    
    Args:
        image: 2D or 3D array representing an image
        new_height: Target height of the resized image
        new_width: Target width of the resized image
    
    Returns:
        Resized image as a nested list with values rounded to 2 decimal places.
    """
    img_arr = np.asarray(image, dtype=np.float64)
    
    # Extract dimensions
    if img_arr.ndim == 2:
        H, W = img_arr.shape
        C = 1
        img_arr = img_arr[:, :, np.newaxis]  # Expand to (H, W, 1) for uniform handling
    elif img_arr.ndim == 3:
        H, W, C = img_arr.shape
    else:
        return []

    scale_r = H / new_height
    scale_c = W / new_width

    # Generate grid coordinates for output image
    i_indices = np.arange(new_height)
    j_indices = np.arange(new_width)
    
    # Calculate mapped coordinates in source image
    r = i_indices * scale_r
    c = j_indices * scale_c
    
    # Compute corner integer coordinates
    r1 = np.floor(r).astype(int)
    c1 = np.floor(c).astype(int)
    
    # Clamp maximum bounds to prevent index overflow
    r2 = np.clip(r1 + 1, 0, H - 1)
    c2 = np.clip(c1 + 1, 0, W - 1)
    
    # Fractional offsets for weights
    dr = (r - r1)[:, np.newaxis]  # Shape (new_height, 1)
    dc = (c - c1)[np.newaxis, :]  # Shape (1, new_width)

    # Output array allocation
    resized = np.zeros((new_height, new_width, C), dtype=np.float64)

    # Vectorized interpolation across 4 grid points
    for k in range(C):
        channel = img_arr[:, :, k]
        
        Q11 = channel[np.ix_(r1, c1)]
        Q12 = channel[np.ix_(r1, c2)]
        Q21 = channel[np.ix_(r2, c1)]
        Q22 = channel[np.ix_(r2, c2)]
        
        resized[:, :, k] = (
            (1 - dr) * (1 - dc) * Q11 +
            (1 - dr) * dc       * Q12 +
            dr       * (1 - dc) * Q21 +
            dr       * dc       * Q22
        )

    # Squeeze extra dimension if original input was 2D
    if C == 1:
        resized = resized.squeeze(axis=2)

    return np.round(resized, 2).tolist()