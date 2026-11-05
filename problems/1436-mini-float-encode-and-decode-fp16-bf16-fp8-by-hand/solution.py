def minifloat_decode(bits, e, m, reserve_top=True):
    """Decode an integer bit pattern of width 1 + e + m to a float."""
    mask_m = (1 << m) - 1
    mask_e = (1 << e) - 1
    
    mantissa = bits & mask_m
    exponent = (bits >> m) & mask_e
    sign = (bits >> (m + e)) & 1
    
    bias = (1 << (e - 1)) - 1
    
    if reserve_top:
        if exponent == mask_e:
            if mantissa == 0:
                return float('-inf') if sign else float('inf')
            else:
                return float('nan')
    else:
        # E4M3 standard single NaN code (sign=1, all ones exponent, all ones mantissa)
        if sign == 1 and exponent == mask_e and mantissa == mask_m:
            return float('nan')
            
    if exponent == 0:
        val = (mantissa / (2.0 ** m)) * (2.0 ** (1 - bias))
    else:
        val = (1.0 + mantissa / (2.0 ** m)) * (2.0 ** (exponent - bias))
        
    return -val if sign else val


def max_normal(e, m, reserve_top=True):
    """The largest finite representable value."""
    max_exp = (1 << e) - 2 if reserve_top else (1 << e) - 1
    bias = (1 << (e - 1)) - 1
    actual_exp = max_exp - bias
    max_mantissa = (1 << m) - 1
    return float((1.0 + max_mantissa / (2.0 ** m)) * (2.0 ** actual_exp))


def min_subnormal(e, m):
    """The smallest positive subnormal value."""
    bias = (1 << (e - 1)) - 1
    return float((1.0 / (2.0 ** m)) * (2.0 ** (1 - bias)))


def minifloat_encode(x, e, m, reserve_top=True):
    """The bit pattern of the finite representable value nearest to x (round-to-even)."""
    if x == 0.0:
        return 0
        
    sign = 1 if x < 0 else 0
    x_abs = abs(x)
    
    max_val = max_normal(e, m, reserve_top)
    if x_abs >= max_val:
        # Saturate to max normal
        max_exp = (1 << e) - 2 if reserve_top else (1 << e) - 1
        max_mant = (1 << m) - 1
        return (sign << (e + m)) | (max_exp << m) | max_mant
        
    bias = (1 << (e - 1)) - 1
    
    # Check subnormal range
    min_sub = min_subnormal(e, m)
    if x_abs < (1.0 / (2.0 ** m)) * (2.0 ** (1 - bias)) * 0.5: # below half of min subnormal -> 0
        return sign << (e + m) # preserving sign of zero if needed, or 0
        
    # Search or calculate nearest representable value
    # A simple exhaustive or analytical rounding to nearest even:
    best_bits = 0
    min_diff = float('inf')
    
    total_codes = 1 << (1 + e + m)
    for bits in range(total_codes):
        val = minifloat_decode(bits, e, m, reserve_top)
        if isinstance(val, float) and (val != val or val == float('inf') or val == float('-inf')):
            continue
        diff = abs(val - x)
        if diff < min_diff:
            min_diff = diff
            best_bits = bits
        elif diff == min_diff:
            # Tie-breaking: choose candidate whose mantissa field is even
            curr_mant = bits & ((1 << m) - 1)
            best_mant = best_bits & ((1 << m) - 1)
            if curr_mant % 2 == 0 and best_mant % 2 != 0:
                best_bits = bits
                
    return best_bits


def relative_error(x, e, m, reserve_top=True):
    """|decode(encode(x)) - x| / |x| rounded to 6 decimals."""
    if x == 0.0:
        return 0.0
    bits = minifloat_encode(x, e, m, reserve_top)
    decoded = minifloat_decode(bits, e, m, reserve_top)
    if not isinstance(decoded, float) or decoded != decoded:
        return float('nan')
    return round(abs(decoded - x) / abs(x), 6)