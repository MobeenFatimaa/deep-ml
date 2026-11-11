import numpy as np


def chunked_linear_scan(a, b, T, h0=None):
    """Chunked scan of h_t = a_t * h_{t-1} + b_t along axis 0 with chunk size T.

    Returns (h with the shape of b, states of shape (n_chunks, ...) = state entering each chunk).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    L = a.shape[0]
    
    if L == 0:
        trailing_shape = b.shape[1:]
        return np.zeros_like(b), np.zeros((0, *trailing_shape), dtype=b.dtype)

    n_chunks = (L + T - 1) // T
    pad_len = n_chunks * T
    trailing_shape = b.shape[1:]

    # Pad a with 1.0 (identity step) and b with 0.0
    a_pad = np.ones((pad_len, *trailing_shape), dtype=a.dtype)
    b_pad = np.zeros((pad_len, *trailing_shape), dtype=b.dtype)

    a_pad[:L] = a
    b_pad[:L] = b

    # Reshape into chunks of size T
    a_chunk = a_pad.reshape(n_chunks, T, *trailing_shape)
    b_chunk = b_pad.reshape(n_chunks, T, *trailing_shape)

    # Compute P_all: cumulative product of a along the within-chunk axis (axis 1)
    P_all = np.cumprod(a_chunk, axis=1)

    # Compute local states ell within each chunk starting from zero state
    ell = np.zeros_like(b_chunk)
    curr = np.zeros((n_chunks, *trailing_shape), dtype=b.dtype)
    for k in range(T):
        curr = a_chunk[:, k] * curr + b_chunk[:, k]
        ell[:, k] = curr

    # Compute carry states S across chunks sequentially
    if h0 is None:
        S = np.zeros(trailing_shape, dtype=b.dtype)
    else:
        S = np.asarray(h0, dtype=b.dtype)

    states = np.zeros((n_chunks, *trailing_shape), dtype=b.dtype)
    for j in range(n_chunks):
        states[j] = S
        P_j = P_all[j, -1]
        ell_j_last = ell[j, -1]
        S = P_j * S + ell_j_last

    # Fix-up: combine local states and carried states to get true states h
    h_chunk = ell + P_all * states[:, np.newaxis, ...]
    h_full = h_chunk.reshape(pad_len, *trailing_shape)
    h = h_full[:L]

    return h, states