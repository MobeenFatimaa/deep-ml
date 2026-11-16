import numpy as np


def simulate_margin(steps, lam, eps, lr, dtype, fixed, beta1=0.9, beta2=0.98):
    """
    Train a scalar margin m (loss softplus(-m)) with AdamW from m = 0.

    fixed=False: gradient q - 1 with q = 1/(1 + e^{-m}) rounded in dtype (exactly 0 past the cliff).
    fixed=True: gradient -e^{-m}/(1 + e^{-m}) in float64.
    Returns {"margins", "final", "peak", "zero_steps", "spikes", "max_drop"}.
    """
    m = 0.0
    m1 = 0.0
    v = 0.0
    
    margins = []
    zero_steps = 0
    spikes = 0
    max_drop = 0.0
    prev_g = 0.0
    
    for t in range(1, steps + 1):
        if not fixed:
            e = np.exp(-m)
            e_dtype = np.asarray(e, dtype=dtype)
            one_dtype = np.asarray(1.0, dtype=dtype)
            q_dtype = one_dtype / (one_dtype + e_dtype)
            gt = float(q_dtype - one_dtype)
        else:
            gt = -np.exp(-m) / (1.0 + np.exp(-m))
            
        if gt == 0.0:
            zero_steps += 1
            
        if t > 1 and prev_g == 0.0 and gt != 0.0:
            spikes += 1
            
        prev_g = gt
        
        # Adam moment updates
        m1 = beta1 * m1 + (1.0 - beta1) * gt
        v = beta2 * v + (1.0 - beta2) * (gt ** 2)
        
        # Bias correction
        m_hat = m1 / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        
        # AdamW update
        step_val = lr * (m_hat / (np.sqrt(v_hat) + eps) + lam * m)
        
        prev_m = m
        m = m - step_val
        
        drop = prev_m - m
        if drop > max_drop:
            max_drop = drop
            
        margins.append(round(float(m), 4))
        
    final = margins[-1]
    peak = max(margins)
    
    return {
        "margins": margins,
        "final": final,
        "peak": peak,
        "zero_steps": zero_steps,
        "spikes": spikes,
        "max_drop": round(max_drop, 4)
    }