def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A list of numerical values
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    if not x:
        return []
    
    min_val = min(x)
    max_val = max(x)
    range_val = max_val - min_val
    
    # Handle the edge case where all values are identical
    if range_val == 0:
        return [0.0] * len(x)
    
    return [(val - min_val) / range_val for val in x]