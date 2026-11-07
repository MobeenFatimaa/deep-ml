import numpy as np


def platt_fit(scores, y, n_iter=100):
    """Fit p = sigmoid(a * s + b) by minimizing mean binary cross-entropy

    using Newton's method starting from a=1, b=0.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=float)
    
    a = 1.0
    b = 0.0
    
    for _ in range(n_iter):
        z = a * s + b
        # Stable sigmoid computation
        p = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
        
        diff = p - y
        g_a = np.sum(diff * s)
        g_b = np.sum(diff)
        g = np.array([g_a, g_b])
        
        w = p * (1.0 - p)
        w = np.maximum(w, 1e-12)
        
        H_aa = np.sum(w * s * s)
        H_ab = np.sum(w * s)
        H_bb = np.sum(w)
        H = np.array([[H_aa, H_ab], [H_ab, H_bb]])
        
        try:
            step = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            break
            
        a += step[0]
        b += step[1]
        
        if np.max(np.abs(step)) < 1e-10:
            break
            
    return float(a), float(b)


def platt_predict(scores, a, b):
    """Compute calibrated probabilities using Platt scaling parameters a and b."""
    s = np.asarray(scores, dtype=float)
    z = a * s + b
    p = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
    return p


def isotonic_fit(scores, y):
    """Pool-Adjacent-Violators algorithm for isotonic regression."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=float)
    
    sorted_idx = np.argsort(s)
    s_sorted = s[sorted_idx]
    y_sorted = y[sorted_idx]
    
    unique_scores = []
    weights = []
    values = []
    
    i = 0
    while i < len(s_sorted):
        curr_score = s_sorted[i]
        j = i
        while j < len(s_sorted) and s_sorted[j] == curr_score:
            j += 1
        chunk_y = y_sorted[i:j]
        unique_scores.append(curr_score)
        weights.append(float(len(chunk_y)))
        values.append(float(np.mean(chunk_y)))
        i = j
        
    block_weights = list(weights)
    block_values = list(values)
    
    idx = 0
    while idx < len(block_values) - 1:
        if block_values[idx] > block_values[idx + 1]:
            w1, v1 = block_weights[idx], block_values[idx]
            w2, v2 = block_weights[idx + 1], block_values[idx + 1]
            w_new = w1 + w2
            v_new = (w1 * v1 + w2 * v2) / w_new
            
            block_weights[idx] = w_new
            block_values[idx] = v_new
            
            del block_weights[idx + 1]
            del block_values[idx + 1]
            
            if idx > 0:
                idx -= 1
        else:
            idx += 1
            
    xs = np.array(unique_scores)
    ys = np.zeros_like(xs, dtype=float)
    
    curr_block = 0
    consumed = 0.0
    for idx_u, w in enumerate(weights):
        ys[idx_u] = block_values[curr_block]
        consumed += w
        if consumed >= block_weights[curr_block] - 1e-9:
            consumed = 0.0
            curr_block += 1
            
    return xs, ys


def isotonic_predict(scores, xs, ys):
    """Predict probabilities using linear interpolation between fitted isotonic points,

    clamped outside the training range.
    """
    s = np.asarray(scores, dtype=float)
    return np.interp(s, xs, ys)


def expected_calibration_error(p, y, n_bins=10):
    """Compute Expected Calibration Error (ECE) using equal-width bins on [0, 1]

    with the last bin closed on both ends.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(p)
    if n == 0:
        return 0.0
        
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    
    for b in range(n_bins):
        if b < n_bins - 1:
            mask = (p >= edges[b]) & (p < edges[b+1])
        else:
            mask = (p >= edges[b]) & (p <= edges[b+1])
            
        nb = np.sum(mask)
        if nb > 0:
            p_bar = np.mean(p[mask])
            y_bar = np.mean(y[mask])
            ece += (nb / n) * abs(p_bar - y_bar)
            
    return float(ece)


def calibration_report(scores, y_fit, scores_test, y_test):
    """Fit Platt Scaling and Isotonic Regression, returning ECE scores for raw,

    Platt-scaled, and isotopically-calibrated predictions rounded to 4 decimals.
    """
    a, b = platt_fit(scores, y_fit)
    p_platt = platt_predict(scores_test, a, b)
    
    xs, ys = isotonic_fit(scores, y_fit)
    p_isotonic = isotonic_predict(scores_test, xs, ys)
    
    ece_raw = expected_calibration_error(np.clip(scores_test, 0, 1), y_test)
    ece_platt = expected_calibration_error(p_platt, y_test)
    ece_isotonic = expected_calibration_error(p_isotonic, y_test)
    
    return {
        'ece_raw': round(float(ece_raw), 4),
        'ece_platt': round(float(ece_platt), 4),
        'ece_isotonic': round(float(ece_isotonic), 4)
    }