def orthogonal_projection(v, L):
    """
    Compute the orthogonal projection of vector v onto line L.

    :param v: The vector to be projected
    :param L: The line vector defining the direction of projection
    :return: List representing the projection of v onto L rounded to three decimal places
    """
    v_dot_L = sum(vi * li for vi, li in zip(v, L))
    L_dot_L = sum(li * li for li in L)
    
    if L_dot_L == 0:
        return [0.0 for _ in L]
    
    scalar = v_dot_L / L_dot_L
    return [round(scalar * li, 3) for li in L]