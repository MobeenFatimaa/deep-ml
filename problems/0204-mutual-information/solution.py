import numpy as np

def mutual_information(joint_prob: list[list[float]]) -> float:
    """
    Compute the mutual information between two random variables.
    
    Args:
        joint_prob: 2D joint probability distribution P(X,Y)
    
    Returns:
        Mutual information I(X;Y)
    """
    P = np.array(joint_prob, dtype=float)
    
    # Compute marginal distributions
    p_x = np.sum(P, axis=1, keepdims=True)  # Shape (num_x, 1)
    p_y = np.sum(P, axis=0, keepdims=True)  # Shape (1, num_y)
    
    # Expected joint distribution under independence: p(x_i) * p(y_j)
    expected = p_x * p_y
    
    # Filter non-zero joint probability entries
    mask = P > 0
    
    # Compute mutual information using natural logarithm
    mi = np.sum(P[mask] * np.log(P[mask] / expected[mask]))
    
    return float(mi)