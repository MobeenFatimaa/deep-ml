import numpy as np

def simplified_gptq(w: np.ndarray, H_diag: np.ndarray, bits: int = 4, blocksize: int = 128):
    """
    Simplified GPTQ: sequential symmetric quant with within-block
    diagonal-Hessian error compensation.

    scale = max|w| / qmax once for the row (qmax=2^(bits-1)-1).
    For each i in each block:
      q_i = clamp(round(w_b[i]/scale), -qmax, qmax)
      w_hat_i = q_i * scale
      e = w_b[i] - w_hat_i
      for j > i in block: w_b[j] -= e * H_diag[i] / (H_diag[j] + 1e-8)

    Returns q (int n,), scale (float), w_hat (float n,)
    """
    w_arr = np.asarray(w, dtype=np.float64)
    H_arr = np.asarray(H_diag, dtype=np.float64)
    
    qmax = (1 << (bits - 1)) - 1
    max_w = np.max(np.abs(w_arr))
    scale = float(max_w / qmax) if max_w != 0.0 else 1.0
    
    w_b = w_arr.copy()
    n = len(w_arr)
    q = np.zeros(n, dtype=np.int32)
    w_hat = np.zeros(n, dtype=np.float64)
    
    # Process in blocks
    for start in range(0, n, blocksize):
        end = min(start + blocksize, n)
        for i in range(start, end):
            # Quantize current weight
            val = w_b[i] / scale
            q_i = int(np.clip(np.round(val), -qmax, qmax))
            q[i] = q_i
            w_hat_i = q_i * scale
            w_hat[i] = w_hat_i
            
            # Compute error
            e = w_b[i] - w_hat_i
            
            # Compensate later weights in the same block
            if e != 0.0 and i + 1 < end:
                j_indices = np.arange(i + 1, end)
                w_b[j_indices] -= e * H_arr[i] / (H_arr[j_indices] + 1e-8)
                
    return q, scale, w_hat