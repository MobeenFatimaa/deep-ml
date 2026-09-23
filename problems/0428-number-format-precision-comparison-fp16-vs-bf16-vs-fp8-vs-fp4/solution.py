import numpy as np

def compare_formats(values):
    """
    Simulate quantizing FP32 values to FP16, BF16, FP8_E4M3, and FP4_E2M1 formats.
    """
    vals = [float(v) for v in values]

    def get_format_specs(exp_bits, mant_bits, has_inf):
        bias = (1 << (exp_bits - 1)) - 1
        
        # Minimum positive normal value: 2^(1 - bias)
        min_pos_normal = float(2.0 ** (1 - bias))
        
        max_exp = (1 << exp_bits) - 1
        
        if has_inf:
            max_norm_exp = max_exp - 1
            max_mant = (1 << mant_bits) - 1
        else:
            max_norm_exp = max_exp
            max_mant = (1 << mant_bits) - 2
            
        max_rep = float((2 ** (max_norm_exp - bias)) * (1.0 + max_mant / (2 ** mant_bits)))
        
        representable = set()
        
        for s in [0, 1]:
            sign = -1.0 if s == 1 else 1.0
            
            # Subnormal numbers
            for m in range(1 << mant_bits):
                val = sign * (2 ** (1 - bias)) * (m / (2 ** mant_bits))
                representable.add((val, 0, m))
                
            # Normal numbers
            max_e = (max_exp - 1) if has_inf else max_exp
            for e in range(1, max_e + 1):
                for m in range(1 << mant_bits):
                    if not has_inf and e == max_exp and m == ((1 << mant_bits) - 1):
                        continue
                    val = sign * (2 ** (e - bias)) * (1.0 + m / (2 ** mant_bits))
                    representable.add((val, e, m))

        sorted_reps = sorted(list(representable), key=lambda x: x[0])
        rep_vals = np.array([r[0] for r in sorted_reps])
        
        return {
            'bias': bias,
            'has_inf': has_inf,
            'max_rep': max_rep,
            'min_pos_normal': min_pos_normal,
            'sorted_reps': sorted_reps,
            'rep_vals': rep_vals
        }

    specs = {
        'fp16': get_format_specs(5, 10, True),
        'bf16': get_format_specs(8, 7, True),
        'fp8_e4m3': get_format_specs(4, 3, False),
        'fp4_e2m1': get_format_specs(2, 1, False)
    }

    results = {}

    for fmt_name, spec in specs.items():
        max_rep = spec['max_rep']
        has_inf = spec['has_inf']
        sorted_reps = spec['sorted_reps']
        rep_vals = spec['rep_vals']
        
        quantized_list = []
        
        for v in vals:
            if np.isnan(v):
                quantized_list.append(float('nan'))
                continue
                
            if v > max_rep:
                quantized_list.append(float('inf') if has_inf else max_rep)
                continue
            elif v < -max_rep:
                quantized_list.append(float('-inf') if has_inf else -max_rep)
                continue
                
            idx = np.searchsorted(rep_vals, v)
            
            if idx == 0:
                best_val = rep_vals[0]
            elif idx == len(rep_vals):
                best_val = rep_vals[-1]
            else:
                left_val, left_e, left_m = sorted_reps[idx - 1]
                right_val, right_e, right_m = sorted_reps[idx]
                
                d_left = abs(v - left_val)
                d_right = abs(v - right_val)
                
                if abs(d_left - d_right) < 1e-12:
                    best_val = left_val if (left_m % 2 == 0) else right_val
                elif d_left < d_right:
                    best_val = left_val
                else:
                    best_val = right_val
                    
            quantized_list.append(best_val)

        abs_errors = []
        for orig, q in zip(vals, quantized_list):
            if abs(q) == float('inf'):
                abs_errors.append(float('inf'))
            else:
                abs_errors.append(abs(orig - q))
                
        max_err = float('inf') if float('inf') in abs_errors else round(max(abs_errors), 6)
        mean_err = float('inf') if float('inf') in abs_errors else round(sum(abs_errors) / len(abs_errors), 6)

        results[fmt_name] = {
            'max_representable': spec['max_rep'],
            'min_positive_normal': spec['min_pos_normal'],
            'quantized': quantized_list,
            'max_abs_error': max_err,
            'mean_abs_error': mean_err
        }

    return results