import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    x = np.asarray(x, dtype=np.float64)
    analytical_grad = np.asarray(analytical_grad, dtype=np.float64)
    
    numerical_grad = np.zeros_like(x, dtype=np.float64)
    
    # Iterate through every dimension of x using a flat indexer
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]
        
        # Center finite difference: f(x + eps) - f(x - eps) / (2 * eps)
        x[idx] = old_val + epsilon
        fx_plus = f(x)
        
        x[idx] = old_val - epsilon
        fx_minus = f(x)
        
        x[idx] = old_val  # Restore original value
        
        numerical_grad[idx] = (fx_plus - fx_minus) / (2.0 * epsilon)
        it.iternext()
        
    # Compute relative error: ||g_num - g_anal|| / (||g_num|| + ||g_anal||)
    diff_norm = np.linalg.norm(numerical_grad - analytical_grad)
    num_norm = np.linalg.norm(numerical_grad)
    anal_norm = np.linalg.norm(analytical_grad)
    
    denom = num_norm + anal_norm
    if denom == 0.0:
        relative_error = 0.0
    else:
        relative_error = diff_norm / denom
        
    return numerical_grad, float(relative_error)