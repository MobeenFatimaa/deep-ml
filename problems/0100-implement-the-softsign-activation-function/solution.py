def softsign(x: float) -> float:
    """
    Implements the Softsign activation function.

    Args:
        x (float): Input value

    Returns:
        float: The Softsign of the input
    """
    return x / (1.0 + abs(x))