def find_treasure(start_x: float) -> float:
    """
    Find the x-coordinate where f(x) = x^4 - 3x^3 + 2 is minimized.

    Returns:
        float: The x-coordinate of the minimum point.
    """
    # Nudge away if starting precisely on the flat inflection point at x = 0
    if abs(start_x) < 1e-5:
        start_x = 0.1
        
    x = start_x
    v = 0.0
    alpha = 0.005  # Learning rate
    beta = 0.9     # Momentum coefficient
    
    # Run gradient descent with momentum to navigate past the saddle/flat spot
    for _ in range(15000):
        grad = 4 * x**3 - 9 * x**2
        v = beta * v - alpha * grad
        x += v
        
    return x