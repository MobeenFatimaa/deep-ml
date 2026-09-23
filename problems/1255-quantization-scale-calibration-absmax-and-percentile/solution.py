import numpy as np

def absmax_scale(x: np.ndarray, bits: int = 8) -> float:
    """Symmetric absmax scale: max|x| / qmax, qmax=2^(bits-1)-1."""
    qmax = (1 << (bits - 1)) - 1
    abs_max = float(np.max(np.abs(x)))
    
    if abs_max == 0.0:
        return 1.0
    
    return float(abs_max / qmax)

def percentile_scale(x: np.ndarray, bits: int = 8, p: float = 99.9) -> float:
    """Symmetric percentile scale on |x|."""
    qmax = (1 << (bits - 1)) - 1
    pct_val = float(np.percentile(np.abs(x), p))
    
    if pct_val == 0.0:
        return 1.0
        
    return float(pct_val / qmax)