import numpy as np

def adam_optimizer(parameter, grad, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using the Adam optimizer.
    Adjusts the learning rate based on the moving averages of the gradient and squared gradient.
    :param parameter: Current parameter value (scalar or numpy array)
    :param grad: Current gradient (scalar or numpy array)
    :param m: First moment estimate
    :param v: Second moment estimate
    :param t: Current timestep (int >= 1)
    :param learning_rate: Learning rate (default=0.001)
    :param beta1: First moment decay rate (default=0.9)
    :param beta2: Second moment decay rate (default=0.999)
    :param epsilon: Small constant for numerical stability (default=1e-8)
    :return: tuple: (updated_parameter, updated_m, updated_v)
    """
    parameter = np.asanyarray(parameter, dtype=float)
    grad = np.asanyarray(grad, dtype=float)
    m = np.asanyarray(m, dtype=float)
    v = np.asanyarray(v, dtype=float)
    
    # 1. Update biased first moment estimate
    m_new = beta1 * m + (1.0 - beta1) * grad
    
    # 2. Update biased second raw moment estimate
    v_new = beta2 * v + (1.0 - beta2) * (grad ** 2)
    
    # 3. Compute bias-corrected first moment estimate
    m_hat = m_new / (1.0 - (beta1 ** t))
    
    # 4. Compute bias-corrected second raw moment estimate
    v_hat = v_new / (1.0 - (beta2 ** t))
    
    # 5. Update parameters
    parameter_new = parameter - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    
    # If inputs were pure scalars, return Python floats for convenience
    if parameter.ndim == 0:
        return float(parameter_new), float(m_new), float(v_new)
        
    return parameter_new, m_new, v_new