import numpy as np

def translate_object(points, tx, ty):
    """
    Apply a 2D translation matrix to a set of points.
    
    Args:
        points: List of [x, y] coordinates
        tx: Translation distance in the x direction
        ty: Translation distance in the y direction
        
    Returns:
        List of [x, y] coordinates after applying the translation
    """
    points_arr = np.asarray(points, dtype=np.float64)
    
    # Translation vector
    translation = np.array([tx, ty], dtype=np.float64)
    
    # Apply translation
    translated_arr = points_arr + translation
    
    return translated_arr.tolist()