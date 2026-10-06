def calculate_brightness(img):
    """
    Calculates the average brightness of a 2D grayscale image matrix.
    
    Args:
        img (list[list[int/float]]): 2D matrix representing pixel values [0, 255]
        
    Returns:
        float: Average brightness rounded to 2 decimal places, or -1 for invalid input.
    """
    # 1. Check if the matrix is empty
    if not img or not isinstance(img, list):
        return -1
    
    if not isinstance(img[0], list) or len(img[0]) == 0:
        return -1
    
    expected_cols = len(img[0])
    total_sum = 0
    total_pixels = 0
    
    # 2. Iterate through rows and validate elements
    for row in img:
        # Check for inconsistent row lengths or non-list row types
        if not isinstance(row, list) or len(row) != expected_cols:
            return -1
        
        for pixel in row:
            # Check type and valid pixel value range [0, 255]
            if not isinstance(pixel, (int, float)) or pixel < 0 or pixel > 255:
                return -1
            
            total_sum += pixel
            total_pixels += 1
            
    # 3. Calculate average brightness rounded to two decimal places
    average_brightness = total_sum / total_pixels
    return round(average_brightness, 2)