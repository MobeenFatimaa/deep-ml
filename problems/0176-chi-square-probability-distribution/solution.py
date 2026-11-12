import math

def chi_square_probability(x, k):
    """
    Calculate the probability density of x in a Chi-square distribution
    with k degrees of freedom.
    """
    if x < 0 or k <= 0:
        return 0.0
    
    # For k < 2, the density is unbounded at x = 0
    if x == 0 and k < 2:
        return float('inf')
        
    try:
        # Formula components
        denom = (2.0 ** (k / 2.0)) * math.gamma(k / 2.0)
        coeff = 1.0 / denom
        term1 = (x ** (k / 2.0 - 1.0))
        term2 = math.exp(-x / 2.0)
        
        pdf = coeff * term1 * term2
    except (OverflowError, ZeroDivisionError):
        pdf = float('inf')
        
    return round(pdf, 3)