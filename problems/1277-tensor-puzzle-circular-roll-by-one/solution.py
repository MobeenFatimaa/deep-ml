import numpy as np

def roll(a: np.ndarray) -> np.ndarray:
    """Circular left shift by one."""
    n = a.shape[0]
    # Generate indices shifted left by 1 with wrap-around using modulo arithmetic
    indices = (np.arange(n) + 1) % n
    return a[indices]