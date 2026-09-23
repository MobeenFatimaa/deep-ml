import numpy as np

def asymmetric_quantize(x: np.ndarray, bits: int = 8):
    """
    Asymmetric uniform quantization.

    Returns q (int array), scale (float), zero_point (int), x_hat (float array).
    Unsigned grid: qmin=0, qmax=2^bits-1.
    """
    x = np.asarray(x, dtype=np.float64)
    x_min, x_max = np.min(x), np.max(x)
    
    qmin = 0
    qmax = (1 << bits) - 1
    
    # Check for degenerate range
    if x_max - x_min < 1e-12:
        q = np.zeros_like(x, dtype=np.int64)
        scale = 1.0
        zero_point = 0
        x_hat = x.copy()
        return q, scale, zero_point, x_hat
    
    # Calculate scale
    scale = float((x_max - x_min) / (qmax - qmin))
    
    # Calculate zero point with clamping
    zp_float = qmin - (x_min / scale)
    zero_point = int(np.clip(np.round(zp_float), qmin, qmax))
    
    # Quantize input vector x
    q_float = (x / scale) + zero_point
    q = np.clip(np.round(q_float), qmin, qmax).astype(np.int64)
    
    # Dequantize back to x_hat
    x_hat = scale * (q - zero_point)
    
    return q, scale, zero_point, x_hat