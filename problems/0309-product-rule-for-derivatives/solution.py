import numpy as np

def product_rule_derivative(f_coeffs: list, g_coeffs: list) -> list:
    """
    Compute the derivative of the product of two polynomials.
    
    Args:
        f_coeffs: Coefficients of polynomial f, where f_coeffs[i] is the coefficient of x^i
        g_coeffs: Coefficients of polynomial g, where g_coeffs[i] is the coefficient of x^i
    
    Returns:
        Coefficients of (f*g)' as a list of floats rounded to 4 decimal places
    """
    if not f_coeffs or not g_coeffs:
        return [0.0]
    
    # Compute derivatives f' and g'
    # f'(x) = sum(i * a_i * x^(i-1))
    f_prime = [i * f_coeffs[i] for i in range(1, len(f_coeffs))]
    if not f_prime:
        f_prime = [0.0]
        
    g_prime = [i * g_coeffs[i] for i in range(1, len(g_coeffs))]
    if not g_prime:
        g_prime = [0.0]
        
    # Product rule: (f * g)' = f' * g + f * g'
    # Polynomial multiplication is equivalent to convolution of coefficient lists
    term1 = np.convolve(f_prime, g_coeffs)
    term2 = np.convolve(f_coeffs, g_prime)
    
    # Pad to the same length for element-wise addition
    max_len = max(len(term1), len(term2))
    term1_padded = np.pad(term1, (0, max_len - len(term1)), 'constant')
    term2_padded = np.pad(term2, (0, max_len - len(term2)), 'constant')
    
    result_coeffs = term1_padded + term2_padded
    
    # Round coefficients to 4 decimal places
    result_coeffs = np.round(result_coeffs, 4)
    
    # Convert back to list and remove trailing zero coefficients
    res_list = result_coeffs.tolist()
    while len(res_list) > 1 and res_list[-1] == 0.0:
        res_list.pop()
        
    if not res_list or (len(res_list) == 1 and res_list[0] == 0.0):
        return [0.0]
        
    return [float(x) for x in res_list]