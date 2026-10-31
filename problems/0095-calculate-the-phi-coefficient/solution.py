import math

def phi_corr(x: list[int], y: list[int]) -> float:
    """
    Calculate the Phi coefficient between two binary variables.

    Args:
    x (list[int]): A list of binary values (0 or 1).
    y (list[int]): A list of binary values (0 or 1).

    Returns:
    float: The Phi coefficient rounded to 4 decimal places.
    """
    n11 = sum(1 for xi, yi in zip(x, y) if xi == 1 and yi == 1)
    n10 = sum(1 for xi, yi in zip(x, y) if xi == 1 and yi == 0)
    n01 = sum(1 for xi, yi in zip(x, y) if xi == 0 and yi == 1)
    n00 = sum(1 for xi, yi in zip(x, y) if xi == 0 and yi == 0)
    
    numerator = (n11 * n00) - (n10 * n01)
    denominator_sq = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    
    if denominator_sq == 0:
        return 0.0
        
    val = numerator / math.sqrt(denominator_sq)
    return round(val, 4)