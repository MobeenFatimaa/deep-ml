import numpy as np

def ptq(weights: dict, calib_X: np.ndarray) -> dict:
    """
    Design and apply a Post-Training Quantization (PTQ) scheme for an MNIST MLP.
    
    Args:
        weights: dict containing weights and biases of the MLP:
                 'W1': (128, 784), 'b1': (128,)
                 'W2': (64, 128),  'b2': (64,)
                 'W3': (10, 64),   'b3': (10,)
        calib_X: (n_calib, 784) calibration batch of real MNIST pixels in [0, 1]
        
    Returns:
        dequant_weights: dict with the same keys, shapes, and types, containing 
                         dequantized weight representations.
    """
    dequant_weights = {}
    
    # We choose an 8-bit per-row uniform symmetric quantization scheme for weights (W1, W2, W3)
    # and per-tensor uniform quantization for biases, ensuring robust preservation of logits.
    qmax = 127.0
    
    for key, w in weights.items():
        w_arr = np.asarray(w, dtype=np.float64)
        
        if w_arr.ndim == 2:
            # Per-row symmetric 8-bit quantization for weight matrices
            num_rows = w_arr.shape[0]
            row_max = np.max(np.abs(w_arr), axis=1, keepdims=True)
            scales = row_max / qmax
            # Avoid division by zero
            scales[scales == 0.0] = 1.0
            
            # Quantize and dequantize
            q = np.clip(np.round(w_arr / scales), -qmax, qmax)
            w_dequant = q * scales
            dequant_weights[key] = w_dequant.astype(w.dtype)
            
        elif w_arr.ndim == 1:
            # Per-tensor symmetric 8-bit quantization for biases
            max_val = np.max(np.abs(w_arr))
            scale = max_val / qmax if max_val != 0.0 else 1.0
            
            q = np.clip(np.round(w_arr / scale), -qmax, qmax)
            w_dequant = q * scale
            dequant_weights[key] = w_dequant.astype(w.dtype)
        else:
            # Fallback for any other shapes
            dequant_weights[key] = w_arr.copy()
            
    return dequant_weights