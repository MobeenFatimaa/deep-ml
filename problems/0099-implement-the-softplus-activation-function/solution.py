import math

def softplus(x: float) -> float:
    """
    Compute the softplus activation function.

    Args:
        x: Input value

    Returns:
        The softplus value log(1 + e^x), rounded to 4 decimal places
    """
    # For very large positive x, exp(x) can overflow, but softplus(x) ≈ x.
    # For very large negative x, exp(x) underflows to 0, making log(1 + exp(x)) ≈ 0.
    if x > 20.0:
        val = x
    elif x < -20.0:
        val = math.log1p(math.exp(x))
    else:
        val = math.log(1.0 + math.exp(x))
        
    return round(val, 4)