import numpy as np

def nag_optimizer(parameter, grad_fn, velocity, learning_rate=0.01, momentum=0.9):
    """
    Update parameters using the Nesterov Accelerated Gradient optimizer.
    Uses a "look-ahead" approach to improve convergence by applying momentum before computing the gradient.
    Args:
        parameter: Current parameter value
        grad_fn: Function that computes the gradient at a given position
        velocity: Current velocity (momentum term)
        learning_rate: Learning rate (default=0.01)
        momentum: Momentum coefficient (default=0.9)
    Returns:
        tuple: (updated_parameter, updated_velocity)
    """
    # Convert inputs to numpy arrays to seamlessly handle both scalar and array inputs
    theta = np.asarray(parameter, dtype=float)
    v = np.asarray(velocity, dtype=float)
    
    # 1. Look-ahead position: theta_lookahead = theta - gamma * v
    # Note: In standard optimization literature, 'gamma' or 'mu' represents the momentum coefficient.
    theta_lookahead = theta - momentum * v
    
    # 2. Gradient at look-ahead position: g = grad_fn(theta_lookahead)
    g = np.asarray(grad_fn(theta_lookahead), dtype=float)
    
    # 3. Velocity update: v_new = gamma * v + eta * g
    # (where eta is learning_rate and gamma is momentum)
    v_new = momentum * v + learning_rate * g
    
    # 4. Parameter update: theta_new = theta - v_new
    theta_new = theta - v_new
    
    # If the original inputs were Python scalars, return standard Python floats/scalars
    if np.isscalar(parameter):
        return float(theta_new), float(v_new)
        
    return theta_new, v_new