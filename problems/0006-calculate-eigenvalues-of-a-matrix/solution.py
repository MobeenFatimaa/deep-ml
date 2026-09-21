import math

def calculate_eigenvalues(matrix: list[list[float | int]]) -> list[float]:
    """
    Calculates the eigenvalues of a 2x2 matrix and returns them sorted 
    from highest to lowest.
    
    Parameters:
    ----------
    matrix : list[list[float | int]]
        A 2x2 matrix represented as a list of list of numbers.
        
    Returns:
    -------
    list[float]
        Eigenvalues sorted in descending order.
    """
    a, b = matrix[0][0], matrix[0][1]
    c, d = matrix[1][0], matrix[1][1]

    # Calculate Trace and Determinant
    trace = a + d
    det = a * d - b * c

    # Characteristic Equation: λ² - trace*λ + det = 0
    # Discriminant: Δ = trace² - 4*det
    discriminant = trace**2 - 4 * det

    if discriminant >= 0:
        # Real eigenvalues
        sqrt_disc = math.sqrt(discriminant)
        λ1 = (trace + sqrt_disc) / 2
        λ2 = (trace - sqrt_disc) / 2
    else:
        # Complex eigenvalues (handled if necessary, using complex math)
        import cmath
        sqrt_disc = cmath.sqrt(discriminant)
        λ1 = (trace + sqrt_disc) / 2
        λ2 = (trace - sqrt_disc) / 2
        # Sort complex numbers by their real part, then imaginary part
        eigenvalues = [λ1, λ2]
        eigenvalues.sort(key=lambda x: (x.real, x.imag), reverse=True)
        return eigenvalues

    # Return sorted from highest to lowest
    return [max(λ1, λ2), min(λ1, λ2)]