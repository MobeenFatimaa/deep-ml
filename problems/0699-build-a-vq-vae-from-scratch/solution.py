import numpy as np

def vq_vae_loss(x: np.ndarray, z_e: np.ndarray, x_recon: np.ndarray, codebook: np.ndarray, beta: float) -> float:
    """
    Compute the VQ-VAE training loss.

    Args:
        x: Original inputs, shape (N, input_dim).
        z_e: Encoder outputs, shape (N, D).
        x_recon: Decoder reconstructions, shape (N, input_dim).
        codebook: Codebook embeddings, shape (K, D).
        beta: Commitment loss weight.

    Returns:
        Total VQ-VAE loss as a single float scalar.
    """
    # Compute squared Euclidean distance between z_e and codebook
    # z_e: (N, D), codebook: (K, D)
    dists = np.sum((z_e[:, None, :] - codebook[None, :, :]) ** 2, axis=2)
    
    # Find nearest codebook entry (lowest index on ties)
    indices = np.argmin(dists, axis=1)
    
    # Build quantized tensor z_q
    z_q = codebook[indices]
    
    # Compute loss components
    recon_loss = np.mean((x - x_recon) ** 2)
    codebook_loss = np.mean((z_e - z_q) ** 2)
    commitment_loss = np.mean((z_e - z_q) ** 2)
    
    total_loss = recon_loss + codebook_loss + beta * commitment_loss
    
    return float(total_loss)