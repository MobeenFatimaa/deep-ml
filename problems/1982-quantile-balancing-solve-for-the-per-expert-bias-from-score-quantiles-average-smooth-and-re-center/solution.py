import numpy as np


def _round4_list(arr):
    return (np.round(np.asarray(arr, dtype=float), 4) + 0.0).tolist()


def qb_route(scores, beta, k):
    scores = np.asarray(scores, dtype=float)
    beta = np.asarray(beta, dtype=float)

    T, N = scores.shape
    if (T * k) % N != 0:
        raise ValueError(f"T * k ({T * k}) must be divisible by N ({N}).")

    C = (T * k) // N

    adj = scores - beta[None, :]  # (T, N)
    expert_ids = np.broadcast_to(np.arange(N), (T, N))
    # Sort ascending by (-adj, expert_id) -> largest adj first, ties by lower expert index
    order = np.lexsort((expert_ids, -adj), axis=1)

    idx = order[:, :k]
    a_t = adj[np.arange(T), order[:, k]]  # (k+1)-th largest adjusted score per token

    margins = scores - a_t[:, None]  # (T, N)
    # Sort descending along token axis (axis=0) for each expert
    sorted_margins = np.sort(margins, axis=0)[::-1]
    beta_local = sorted_margins[C, :]  # (C+1)-th largest (0-indexed C)

    return idx, beta_local


def qb_train(steps, k, ema=0.0):
    N = np.asarray(steps[0][0], dtype=float).shape[1]
    beta = np.zeros(N, dtype=float)

    global_maxvio = []
    micro_maxvio = []

    for global_batch in steps:
        total_counts = np.zeros(N, dtype=float)
        micro_betas = []
        micro_mvs = []

        for mb in global_batch:
            mb_arr = np.asarray(mb, dtype=float)
            idx, beta_local = qb_route(mb_arr, beta, k)

            counts = np.bincount(idx.ravel(), minlength=N).astype(float)
            total_counts += counts

            c_bar = np.mean(counts)
            mv = float(np.max(counts / c_bar) - 1.0) if c_bar > 0 else 0.0

            micro_betas.append(beta_local)
            micro_mvs.append(mv)

        beta_bar = np.mean(micro_betas, axis=0)
        beta = ema * beta + (1.0 - ema) * beta_bar
        beta = beta - np.mean(beta)

        g_c_bar = np.mean(total_counts)
        g_mv = float(np.max(total_counts / g_c_bar) - 1.0) if g_c_bar > 0 else 0.0

        global_maxvio.append(round(g_mv, 4))
        micro_maxvio.append(round(float(np.mean(micro_mvs)), 4))

    return {
        "global_maxvio": global_maxvio,
        "micro_maxvio": micro_maxvio,
        "beta": _round4_list(beta),
    }