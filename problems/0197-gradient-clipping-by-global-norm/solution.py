import math

def clip_gradients_by_global_norm(gradients: list[list[float]], max_norm: float) -> list[list[float]]:
    """
    Clip gradients by global norm.
    
    Args:
        gradients: List of gradient arrays
        max_norm: Maximum allowed global norm
    
    Returns:
        List of clipped gradient arrays
    """
    # 1. Compute the sum of squared values across all gradient arrays
    total_norm_sq = sum(val ** 2 for grad in gradients for val in grad)
    
    # 2. Calculate the global L2 norm
    global_norm = math.sqrt(total_norm_sq)
    
    # 3. If global norm exceeds max_norm, compute the scaling factor and scale down
    if global_norm > max_norm and global_norm > 0:
        scale_factor = max_norm / global_norm
        return [[val * scale_factor for val in grad] for grad in gradients]
    
    # Otherwise, return a copy of the original gradients
    return [[val for val in grad] for grad in gradients]