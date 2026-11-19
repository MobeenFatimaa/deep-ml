import numpy as np


# ---- provided: the original model and the JumpReLU
def jumprelu(pre, theta):
    return np.where(pre > theta, pre, 0.0)


def original_forward(h0, model):
    h = np.array(h0, dtype=float)
    T, d = h.shape
    mask = np.triu(np.ones((T, T), dtype=bool), 1)
    run = {"A": [], "s": [], "x": [], "mlp_out": []}
    for l in range(len(model["W_OV"])):
        sc = np.where(mask, -np.inf, h @ model["W_QK"][l] @ h.T / np.sqrt(d))
        A = np.exp(sc - sc.max(axis=1, keepdims=True))
        A = A / A.sum(axis=1, keepdims=True)
        mid = h + A @ h @ model["W_OV"][l]
        s = np.sqrt((mid ** 2).mean(axis=1) + 1e-6)
        x = mid / s[:, None]
        out = np.maximum(x @ model["W1"][l], 0.0) @ model["W2"][l]
        h = mid + out
        run["A"].append(A)
        run["s"].append(s)
        run["x"].append(x)
        run["mlp_out"].append(out)
    run["h"] = h
    return run


# ---- end provided


def error_terms(orig, clt):
    """Per-layer orig["mlp_out"] minus the CLT reconstruction computed from orig["x"]."""
    L = len(orig["mlp_out"])
    recon = [np.zeros_like(orig["mlp_out"][lp], dtype=float) for lp in range(L)]

    for l in range(L):
        z_l = jumprelu(orig["x"][l] @ clt["W_enc"][l], clt["theta"][l])
        for lp in range(l, L):
            w_dec = clt["W_dec"][l][lp]
            if w_dec is not None:
                recon[lp] = recon[lp] + z_l @ w_dec

    return [orig["mlp_out"][lp] - recon[lp] for lp in range(L)]


def replacement_forward(h0, model, clt, frozen, errors=None):
    """Frozen attention and scales, MLPs replaced by CLT reconstructions (+ errors). Returns x, z, mlp_out, h."""
    h = np.array(h0, dtype=float)
    L = len(model["W_OV"])

    pending_writes = [np.zeros_like(h, dtype=float) for _ in range(L)]
    x_list = []
    z_list = []
    mlp_out_list = []

    for l in range(L):
        A = frozen["A"][l]
        s = frozen["s"][l]

        mid = h + A @ h @ model["W_OV"][l]
        x = mid / s[:, None]
        z_l = jumprelu(x @ clt["W_enc"][l], clt["theta"][l])

        for lp in range(l, L):
            w_dec = clt["W_dec"][l][lp]
            if w_dec is not None:
                pending_writes[lp] = pending_writes[lp] + z_l @ w_dec

        err_l = errors[l] if errors is not None else 0.0
        out = pending_writes[l] + err_l
        h = mid + out

        x_list.append(x)
        z_list.append(z_l)
        mlp_out_list.append(out)

    return {
        "x": x_list,
        "z": z_list,
        "mlp_out": mlp_out_list,
        "h": h,
    }


def direct_edge(model, clt, frozen, z, source, target):
    """Direct effect of source feature (ls, cs, js) on target pre-activation (lt, ct, jt)."""
    ls, cs, js = source
    lt, ct, jt = target

    if lt <= ls:
        return 0.0

    z_val = float(z[ls][cs, js])
    if z_val == 0.0:
        return 0.0

    T, d = frozen["x"][ls].shape
    v_in_t = clt["W_enc"][lt][:, jt]
    s_t = frozen["s"][lt][ct]
    total = 0.0

    for ell in range(ls, lt):
        w_dec = clt["W_dec"][ls][ell]
        if w_dec is None:
            continue

        delta = np.zeros((T, d), dtype=float)
        delta[cs] = w_dec[js]

        for k in range(ell + 1, lt + 1):
            delta = delta + frozen["A"][k] @ delta @ model["W_OV"][k]

        total += z_val * float((delta[ct] @ v_in_t) / s_t)

    return float(total)