import numpy as np

def video_tokenize(video, codebook, encoder_weight, patch_size):
    """
    Tokenize a video into a grid of discrete codebook indices using a
    spatiotemporal VQ-VAE encoder.

    Args:
        video: array of shape (T, H, W, C).
        codebook: array of shape (K, D).
        encoder_weight: array of shape (pt*ph*pw*C, D).
        patch_size: tuple (pt, ph, pw).

    Returns:
        Nested list of shape (T/pt, H/ph, W/pw) with codebook indices,
        or -1 if the inputs are invalid.
    """
    try:
        video = np.array(video, dtype=np.float32)
        codebook = np.array(codebook, dtype=np.float32)
        encoder_weight = np.array(encoder_weight, dtype=np.float32)
    except Exception:
        return -1

    if video.ndim != 4 or codebook.ndim != 2 or encoder_weight.ndim != 2:
        return -1

    if not isinstance(patch_size, (list, tuple)) or len(patch_size) != 3:
        return -1

    T, H, W, C = video.shape
    pt, ph, pw = patch_size

    if pt <= 0 or ph <= 0 or pw <= 0:
        return -1
    if T % pt != 0 or H % ph != 0 or W % pw != 0:
        return -1

    K, D = codebook.shape
    expected_weight_dim = pt * ph * pw * C
    if encoder_weight.shape[0] != expected_weight_dim or encoder_weight.shape[1] != D:
        return -1

    nT = T // pt
    nH = H // ph
    nW = W // pw

    result = []
    for it in range(nT):
        t_start = it * pt
        t_end = (it + 1) * pt
        row = []
        for ih in range(nH):
            h_start = ih * ph
            h_end = (ih + 1) * ph
            col = []
            for iw in range(nW):
                w_start = iw * pw
                w_end = (iw + 1) * pw
                
                patch = video[t_start:t_end, h_start:h_end, w_start:w_end, :]
                flat_patch = patch.flatten()
                latent = flat_patch @ encoder_weight
                
                diff = codebook - latent
                dist = np.sum(diff ** 2, axis=1)
                
                min_idx = int(np.argmin(dist))
                col.append(min_idx)
            row.append(col)
        result.append(row)

    return result