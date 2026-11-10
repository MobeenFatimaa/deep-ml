import numpy as np

def spike_steps(values, num_samples=100, factor=10.0):
    values = list(values)
    ref = []
    flagged = []
    for i, v in enumerate(values):
        if not np.isfinite(v):
            continue
        if len(ref) < num_samples:
            ref.append(abs(v))
        else:
            m = max(ref) if ref else 0.0
            if abs(v) >= factor * m:
                flagged.append(i)
    return flagged

def loss_alarm_steps(losses, margin=0.1):
    losses = list(losses)
    flagged = []
    best_loss = float('inf')
    for i, l in enumerate(losses):
        if not np.isfinite(l):
            continue
        if l < best_loss:
            best_loss = l
        if l > best_loss + margin:
            flagged.append(i)
    return flagged

def maxvio(counts):
    counts = np.array(counts, dtype=float)
    if counts.size == 0 or np.all(counts == 0):
        return None
    max_c = np.max(counts)
    mean_c = np.mean(counts)
    if mean_c == 0:
        return None
    return float(max_c / mean_c - 1.0)

def expert_norm_ratios(norm_sum, counts):
    norm_sum = np.array(norm_sum, dtype=float)
    counts = np.array(counts, dtype=float)
    
    empty = [int(e) for e in np.where(counts == 0)[0]]
    empty.sort()
    
    valid_mask = counts > 0
    if not np.any(valid_mask):
        return {"max_ratio": None, "min_ratio": None, "argmax": None, "argmin": None, "empty": empty}
    
    means = norm_sum[valid_mask] / counts[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    n_tilde = np.median(means)
    if n_tilde == 0 or not np.isfinite(n_tilde):
        return {"max_ratio": None, "min_ratio": None, "argmax": None, "argmin": None, "empty": empty}
    
    ratios = means / n_tilde
    
    max_val = np.max(ratios)
    min_val = np.min(ratios)
    
    max_indices = valid_indices[np.isclose(ratios, max_val, atol=1e-9, rtol=1e-9)]
    min_indices = valid_indices[np.isclose(ratios, min_val, atol=1e-9, rtol=1e-9)]
    
    argmax = int(np.min(max_indices))
    argmin = int(np.min(min_indices))
    
    return {
        "max_ratio": float(np.round(max_val, 4)),
        "min_ratio": float(np.round(min_val, 4)),
        "argmax": argmax,
        "argmin": argmin,
        "empty": empty
    }

def health_report(loss, grad_norm, counts, norm_sum, maxvio_limit=1.0, rogue_ratio=10.0, dead_ratio=0.1, margin=0.1, num_samples=100, factor=10.0):
    loss = np.array(loss, dtype=float)
    grad_norm = np.array(grad_norm, dtype=float)
    counts = np.array(counts)
    norm_sum = np.array(norm_sum, dtype=float)
    
    S = len(loss)
    nonfinite = [s for s in range(S) if not np.isfinite(loss[s]) or not np.isfinite(grad_norm[s])]
            
    loss_spikes = spike_steps(loss, num_samples=num_samples, factor=factor)
    grad_spikes = spike_steps(grad_norm, num_samples=num_samples, factor=factor)
    loss_alarm = loss_alarm_steps(loss, margin=margin)
    
    collapse = []
    rogue = []
    dead = []
    empty_list = []
    
    S_dim, L_dim, E_dim = counts.shape
    for s in range(S_dim):
        for l in range(L_dim):
            c_sl = counts[s, l]
            ns_sl = norm_sum[s, l]
            
            mv = maxvio(c_sl)
            if mv is not None and mv > maxvio_limit:
                collapse.append([s, l])
                
            res = expert_norm_ratios(ns_sl, c_sl)
            max_ratio = res["max_ratio"]
            min_ratio = res["min_ratio"]
            
            if max_ratio is not None and max_ratio > rogue_ratio:
                rogue.append([s, l, res["argmax"]])
                
            if min_ratio is not None and min_ratio < dead_ratio:
                dead.append([s, l, res["argmin"]])
                
            for e in res["empty"]:
                empty_list.append([s, l, e])
                
    return {
        "nonfinite": nonfinite,
        "loss_spikes": loss_spikes,
        "grad_spikes": grad_spikes,
        "loss_alarm": loss_alarm,
        "collapse": collapse,
        "rogue": rogue,
        "dead": dead,
        "empty": empty_list
    }