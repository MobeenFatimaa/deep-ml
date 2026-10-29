import numpy as np

def video_tokenizer_reconstruct(video, patch_t: int, patch_h: int, patch_w: int, mode: str):
    """
    Reconstruct a video after mean-pool tokenization.

    Args:
        video: array-like of shape (T, H, W, C)
        patch_t: temporal patch size (ignored when mode == 'spatial')
        patch_h: spatial patch height
        patch_w: spatial patch width
        mode: 'spatial' or 'spatiotemporal'

    Returns:
        Reconstructed video as a nested list of shape (T, H, W, C).
    """
    video = np.array(video, dtype=float)
    T, H, W, C = video.shape
    
    if mode == 'spatial':
        pt = 1
    elif mode == 'spatiotemporal':
        pt = patch_t
    else:
        raise ValueError("mode must be either 'spatial' or 'spatiotemporal'")
        
    ph = patch_h
    pw = patch_w
    
    reconstructed = np.zeros_like(video)
    
    n_t = T // pt
    n_h = H // ph
    n_w = W // pw
    
    for t_idx in range(n_t):
        for h_idx in range(n_h):
            for w_idx in range(n_w):
                t_start = t_idx * pt
                t_end = (t_idx + 1) * pt
                h_start = h_idx * ph
                h_end = (h_idx + 1) * ph
                w_start = w_idx * pw
                w_end = (w_idx + 1) * pw
                
                patch = video[t_start:t_end, h_start:h_end, w_start:w_end, :]
                mean_val = np.mean(patch, axis=(0, 1, 2), keepdims=True)
                reconstructed[t_start:t_end, h_start:h_end, w_start:w_end, :] = mean_val
                
    return reconstructed.tolist()