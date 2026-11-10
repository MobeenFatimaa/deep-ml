import numpy as np

def adamw_update(w, g, m, v, t, lr, beta1, beta2, epsilon, weight_decay):
    """
    Perform one AdamW optimizer step.
    Args:
      w: parameter vector (np.ndarray)
      g: gradient vector (np.ndarray)
      m: first moment vector (np.ndarray)
      v: second moment vector (np.ndarray)
      t: integer, current time step
      lr: float, learning rate
      beta1: float, beta1 parameter
      beta2: float, beta2 parameter
      epsilon: float, small constant
      weight_decay: float, weight decay coefficient
    Returns:
      w_new, m_new, v_new
    """
    w = np.asanyarray(w, dtype=float)
    g = np.asanyarray(g, dtype=float)
    m = np.asanyarray(m, dtype=float)
    v = np.asanyarray(v, dtype=float)
    
    # 1. Update biased first and second moments
    m_new = beta1 * m + (1.0 - beta1) * g
    v_new = beta2 * v + (1.0 - beta2) * (g ** 2)
    
    # 2. Compute bias-corrected estimates
    m_hat = m_new / (1.0 - (beta1 ** t))
    v_hat = v_new / (1.0 - (beta2 ** t))
    
    # 3. Apply decoupled weight decay directly to weights
    w_decayed = w - lr * weight_decay * w
    
    # 4. Apply Adam gradient step on the decayed weights
    w_new = w_decayed - lr * m_hat / (np.sqrt(v_hat) + epsilon)
    
    return w_new, m_new, v_new