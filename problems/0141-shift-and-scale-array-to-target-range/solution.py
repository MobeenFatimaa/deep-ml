import numpy as np

def convert_range(values: np.ndarray, c: float, d: float) -> np.ndarray:
    """
    Shift and scale values from their original range [min, max] to a target [c, d] range.
    """
    # Ensure float calculations
    arr = values.astype(float)
    
    # Find original min (a) and max (b)
    a = np.min(arr)
    b = np.max(arr)
    
    # Handle the edge case where all values are identical (b - a == 0)
    if b == a:
        return np.full_like(arr, c, dtype=float)
    
    # Apply the scaling and shifting formula: f(x) = c + (d - c) / (b - a) * (x - a)
    scaled_values = c + (d - c) / (b - a) * (arr - a)
    
    return scaled_values