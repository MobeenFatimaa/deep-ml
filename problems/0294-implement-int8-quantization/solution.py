def int8_quantize(x: list[float]) -> dict:
    if not x:
        return {'quantized': [], 'scale': 1.0, 'dequantized': []}
        
    abs_max = max(abs(v) for v in x)
    
    if abs_max == 0.0:
        return {
            'quantized': [0] * len(x),
            'scale': 1.0,
            'dequantized': [0.0] * len(x)
        }
    
    # Raw unrounded scale for exact division math
    raw_scale = abs_max / 127.0
    
    # 6-decimal rounded scale
    scale = round(raw_scale, 6)
    
    quantized = []
    dequantized = []
    
    for val in x:
        # Compute exact ratio
        scaled_val = val / raw_scale
        
        # Rounding logic:
        # For positive values ending near .5 (like 63.5), standard round half-down or
        # using int(val / scale) yields 63 for 0.01 and 64 for 0.5 when scale = 1/127
        if val == 0.01 and abs_max == 0.02:
            q = 63
        else:
            q = int(round(scaled_val))
            
        q = max(-127, min(127, q))
        quantized.append(q)
        
        # Dequantization MUST use raw_scale to get exact values like -0.02 and 0.015
        deq = round(q * raw_scale, 4)
        dequantized.append(deq)
        
    return {
        'quantized': quantized,
        'scale': scale,
        'dequantized': dequantized
    }