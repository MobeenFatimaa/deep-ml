import math

def selu(x: float) -> float:
    """
    Implements the SELU (Scaled Exponential Linear Unit) activation function.

    Args:
        x: Input value

    Returns:
        SELU activation value rounded to 4 decimal places
    """
    alpha = 1.6732632423543772
    scale = 1.0507009873554804
    
    if x > 0.0:
        val = scale * x
    else:
        val = scale * alpha * (math.exp(x) - 1.0)
        
    return round(val, 4)