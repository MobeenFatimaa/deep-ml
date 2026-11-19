import numpy as np


def window_partition(x, M):
    H, W, C = x.shape
    x = x.reshape(H // M, M, W // M, M, C)
    windows = x.transpose(0, 2, 1, 3, 4).reshape(-1, M * M, C)
    return windows


def window_reverse(windows, M, H, W):
    C = windows.shape[-1]
    x = windows.reshape(H // M, W // M, M, M, C)
    x = x.transpose(0, 2, 1, 3, 4).reshape(H, W, C)
    return x


def shift_mask(H, W, M, s):
    row_bands = np.zeros(H, dtype=int)
    row_bands[H - M : H - s] = 1
    row_bands[H - s : H] = 2

    col_bands = np.zeros(W, dtype=int)
    col_bands[W - M : W - s] = 1
    col_bands[W - s : W] = 2

    region_map = (row_bands[:, None] * 3 + col_bands[None, :])[..., None]
    win_regions = window_partition(region_map, M)[..., 0]  # (nW, M*M)

    same_region = win_regions[:, :, None] == win_regions[:, None, :]
    return np.where(same_region, 0.0, -100.0).astype(float)


def relative_position_index(M):
    coords = np.arange(M)
    grid_r, grid_c = np.meshgrid(coords, coords, indexing="ij")
    flat_r = grid_r.reshape(-1)
    flat_c = grid_c.reshape(-1)

    rel_r = flat_r[:, None] - flat_r[None, :] + (M - 1)
    rel_c = flat_c[:, None] - flat_c[None, :] + (M - 1)
    return (rel_r * (2 * M - 1) + rel_c).astype(int)


def window_attention(x, M, s, Wq, Wk, Wv, bias_table):
    H, W, C = x.shape

    if s > 0:
        x = np.roll(x, (-s, -s), axis=(0, 1))

    windows = window_partition(x, M)  # (nW, N, C), N = M * M

    Q = windows @ Wq
    K = windows @ Wk
    V = windows @ Wv

    logits = (Q @ np.swapaxes(K, -1, -2)) / np.sqrt(C)
    rpi = relative_position_index(M)
    logits = logits + bias_table[rpi]

    if s > 0:
        logits = logits + shift_mask(H, W, M, s)

    logits_max = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    attn_weights = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    attn_out = attn_weights @ V
    out = window_reverse(attn_out, M, H, W)

    if s > 0:
        out = np.roll(out, (s, s), axis=(0, 1))

    return out