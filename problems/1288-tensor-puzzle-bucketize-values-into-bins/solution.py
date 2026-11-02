import numpy as np

def bucketize(v: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Return bucket indices for v given sorted boundaries."""
    # Broadcast v and boundaries to compare every value against every boundary:
    # v[:, None] has shape (len(v), 1)
    # boundaries[None, :] has shape (1, len(boundaries))
    # The comparison results in a boolean matrix of shape (len(v), len(boundaries))
    comparisons = v[:, None] >= boundaries[None, :]
    
    # Summing across the boundary dimension (axis=1) counts how many 
    # boundaries each value is greater than or equal to, giving the exact bucket index.
    return np.sum(comparisons, axis=1)