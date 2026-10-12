import numpy as np

def compare_input_modalities(prev_frames: np.ndarray, next_frames: np.ndarray, codebook: np.ndarray) -> dict:
    """
    Compare pixel-input vs token-input latent action reconstruction.
    Returns a dict with keys: 'pixel_mse', 'token_mse', 'winner'.
    """
    # 1. Pixel Modality
    pixel_encoded_prev = prev_frames
    pixel_encoded_next = next_frames
    pixel_action = np.round(pixel_encoded_next - pixel_encoded_prev)
    pixel_reconstructed = pixel_encoded_prev + pixel_action
    pixel_mse = float(np.mean((pixel_reconstructed - next_frames) ** 2))

    # 2. Token Modality
    # Find nearest codebook vector for previous frames
    dists_prev = np.sum((prev_frames[:, None, :] - codebook[None, :, :]) ** 2, axis=2)
    prev_indices = np.argmin(dists_prev, axis=1)
    token_encoded_prev = codebook[prev_indices]

    # Find nearest codebook vector for next frames
    dists_next = np.sum((next_frames[:, None, :] - codebook[None, :, :]) ** 2, axis=2)
    next_indices = np.argmin(dists_next, axis=1)
    token_encoded_next = codebook[next_indices]

    token_action = np.round(token_encoded_next - token_encoded_prev)
    token_reconstructed = token_encoded_prev + token_action
    token_mse = float(np.mean((token_reconstructed - next_frames) ** 2))

    # 3. Determine Winner
    if pixel_mse < token_mse:
        winner = "pixel"
    elif token_mse < pixel_mse:
        winner = "token"
    else:
        winner = "tie"

    return {
        'pixel_mse': pixel_mse,
        'token_mse': token_mse,
        'winner': winner
    }