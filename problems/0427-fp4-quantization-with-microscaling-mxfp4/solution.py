import numpy as np

def mxfp4_quantize(x: list, block_size: int = 4) -> dict:
    """
    Perform MXFP4 quantization with per-block microscaling.

    Args:
        x: list of float values to quantize
        block_size: number of elements per scaling block

    Returns:
        dict with keys:
            'quantized': list of dequantized values (rounded to 4 decimals)
            'scales': list of per-block scale factors (rounded to 4 decimals)
    """
    fp4_vals = np.array([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    
    x_arr = np.array(x, dtype=np.float64)
    n = len(x_arr)
    
    # Pad to block_size if needed
    n_blocks = (n + block_size - 1) // block_size
    padded_len = n_blocks * block_size
    if padded_len > n:
        x_padded = np.pad(x_arr, (0, padded_len - n), mode='constant', constant_values=0.0)
    else:
        x_padded = x_arr
        
    quantized_result = []
    scales_result = []
    
    for i in range(n_blocks):
        block = x_padded[i * block_size : (i + 1) * block_size]
        amax = np.max(np.abs(block))
        
        if amax == 0.0:
            scale = 1.0
        else:
            raw_scale = amax / 6.0
            scale = float(2.0 ** np.ceil(np.log2(raw_scale)))
            
        scales_result.append(round(scale, 4))
        
        # Scale block
        scaled_block = block / scale
        
        # Quantize each element
        quantized_block = []
        for val in scaled_block:
            diffs = np.abs(fp4_vals - val)
            min_diff = np.min(diffs)
            candidates = fp4_vals[np.isclose(diffs, min_diff)]
            # Tie breaking: choose candidate with smaller absolute magnitude
            best_candidate = candidates[np.argmin(np.abs(candidates))]
            
            # Dequantize
            dequant_val = best_candidate * scale
            quantized_block.append(dequant_val)
            
        quantized_result.extend(quantized_block)
        
    # Trim to original length and round
    quantized_result = [round(float(v), 4) for v in quantized_result[:n]]
    scales_result = [round(float(s), 4) for s in scales_result]
    
    return {
        "quantized": quantized_result,
        "scales": scales_result
    }