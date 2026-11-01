import numpy as np

def adadelta_optimizer(parameter, grad, u, v, rho=0.95, epsilon=1e-6):
    """
    Update parameters using the AdaDelta optimizer.
    AdaDelta is an extension of AdaGrad that seeks to reduce its aggressive,
    monotonically decreasing learning rate.
    Args:
        parameter: Current parameter value
        grad: Current gradient
        u: Running average of squared gradients
        v: Running average of squared parameter updates
        rho: Decay rate for the moving average (default=0.95)
        epsilon: Small constant for numerical stability (default=1e-6)
    Returns:
        tuple: (updated_parameter, updated_u, updated_v)
    """
    is_scalar = not any(isinstance(arg, np.ndarray) for arg in (parameter, grad, u, v))
    
    param_arr = np.asarray(parameter, dtype=float)
    grad_arr = np.asarray(grad, dtype=float)
    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    
    # 1. Update running average of squared gradients
    u_new = rho * u_arr + (1.0 - rho) * (grad_arr ** 2)
    
    # 2. Compute parameter update delta based on RMS of past updates and current gradients
    delta = - (np.sqrt(v_arr + epsilon) / np.sqrt(u_new + epsilon)) * grad_arr
    
    # 3. Apply update to parameters
    parameter_updated = param_arr + delta
    
    # 4. Update running average of squared parameter updates
    v_updated = rho * v_arr + (1.0 - rho) * (delta ** 2)
    u_updated = u_new
    
    if is_scalar:
        return float(np.round(parameter_updated.item(), 5)), float(np.round(u_updated.item(), 5)), float(np.round(v_updated.item(), 5))
        
    return np.round(parameter_updated, 5), np.round(u_updated, 5), np.round(v_updated, 5)