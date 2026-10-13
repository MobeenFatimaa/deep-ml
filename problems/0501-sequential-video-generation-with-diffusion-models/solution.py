import numpy as np

def sequential_video_diffusion(
    noise_chunks: np.ndarray,
    Wd: np.ndarray,
    alpha_bar: np.ndarray,
    n_context: int,
    n_overlap: int,
) -> np.ndarray:
    """
    Generate video frames sequentially using chunk-based diffusion denoising.

    Args:
        noise_chunks: Initial noise, shape (n_chunks, chunk_frames, H, W)
        Wd: Denoiser weight matrix, shape (H*W, H*W)
        alpha_bar: Cumulative noise schedule, shape (n_steps+1,)
        n_context: Number of context frames from previous chunk
        n_overlap: Number of frames to blend between consecutive chunks

    Returns:
        Generated video, shape (total_frames, H, W)
    """
    n_chunks, chunk_frames, H, W = noise_chunks.shape
    n_steps = len(alpha_bar) - 1
    D = H * W
    generated_chunks = []

    context_bias = np.zeros(D, dtype=float)

    for c in range(n_chunks):
        chunk = noise_chunks[c].copy()
        
        # Iterative denoising from t = n_steps down to 1
        for t in range(n_steps, 0, -1):
            alpha_t = alpha_bar[t]
            alpha_prev = alpha_bar[t - 1]
            
            sqrt_alpha_t = np.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = np.sqrt(1.0 - alpha_t)
            sqrt_alpha_prev = np.sqrt(alpha_prev)
            sqrt_one_minus_alpha_prev = np.sqrt(1.0 - alpha_prev)
            
            new_chunk = np.zeros_like(chunk)
            for f_idx in range(chunk_frames):
                x = chunk[f_idx]
                x_flat = x.reshape(D)
                
                # Predict noise using denoiser weight matrix + context bias
                eps_pred = Wd @ x_flat + context_bias
                
                # Deterministic DDIM-style update
                x0_flat = (x_flat - sqrt_one_minus_alpha_t * eps_pred) / sqrt_alpha_t
                x_prev_flat = sqrt_alpha_prev * x0_flat + sqrt_one_minus_alpha_prev * eps_pred
                
                new_chunk[f_idx] = x_prev_flat.reshape(H, W)
                
            chunk = new_chunk
            
        generated_chunks.append(chunk)
        
        # Compute context bias for the next chunk from the tail of the current generated chunk
        if n_context > 0:
            tail_frames = chunk[-n_context:]
            tail_flat = tail_frames.reshape(n_context, D)
            context_bias = np.mean(tail_flat, axis=0)
        else:
            context_bias = np.zeros(D, dtype=float)

    # Assemble chunks with linear blending over overlapping frames
    video = generated_chunks[0]
    for B in generated_chunks[1:]:
        if n_overlap == 0:
            video = np.concatenate([video, B], axis=0)
        else:
            non_overlap_video = video[:-n_overlap]
            overlap_video = video[-n_overlap:]
            overlap_B = B[:n_overlap]
            
            blended_overlap = np.zeros_like(overlap_video)
            for i in range(n_overlap):
                w_new = (i + 1.0) / (n_overlap + 1.0)
                w_old = 1.0 - w_new
                blended_overlap[i] = w_old * overlap_video[i] + w_new * overlap_B[i]
                
            remaining_B = B[n_overlap:]
            video = np.concatenate([non_overlap_video, blended_overlap, remaining_B], axis=0)

    return video