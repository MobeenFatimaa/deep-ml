import numpy as np

def latent_action_forward(obs_t: np.ndarray,
                         obs_tp1: np.ndarray,
                         W_enc: np.ndarray,
                         b_enc: np.ndarray,
                         codebook: np.ndarray,
                         W_dec: np.ndarray,
                         b_dec: np.ndarray) -> np.ndarray:
    """
    Forward pass for unsupervised latent action discovery.

    Encodes (o_t, o_{t+1}) pairs into a continuous latent, snaps the latent
    to the nearest codebook vector (the discovered discrete action), and
    decodes the predicted next observation from (o_t, z_q).

    Returns:
        Predicted next observations of shape (B, D_obs).
    """
    # 1. Concatenate obs_t and obs_tp1 along the feature axis -> shape (B, 2 * D_obs)
    concatenated_obs = np.concatenate([obs_t, obs_tp1], axis=1)
    
    # 2. Inverse-dynamics encoder linear layer: z_e = concatenated_obs @ W_enc + b_enc
    z_e = concatenated_obs @ W_enc + b_enc  # shape (B, D_latent)
    
    # 3. Discrete bottleneck via vector quantization
    # Compute squared Euclidean distances between each z_e and codebook entries.
    # ||z_e - c||^2 = ||z_e||^2 + ||c||^2 - 2 * z_e @ c^T
    z_e_sq = np.sum(z_e ** 2, axis=1, keepdims=True)  # (B, 1)
    cb_sq = np.sum(codebook ** 2, axis=1)             # (K,)
    cross_term = z_e @ codebook.T                     # (B, K)
    
    distances = z_e_sq + cb_sq - 2.0 * cross_term      # (B, K)
    
    # Find the nearest codebook index for each sample.
    # Using np.argmin with lowest index tie-breaking (argmin returns first occurrence on ties).
    codebook_indices = np.argmin(distances, axis=1)    # (B,)
    z_q = codebook[codebook_indices]                  # (B, D_latent)
    
    # 4. Forward-dynamics decoder: concatenate obs_t and z_q -> shape (B, D_obs + D_latent)
    decoder_input = np.concatenate([obs_t, z_q], axis=1)
    
    # Predict next observation: o_hat_{t+1} = decoder_input @ W_dec + b_dec
    pred_obs_tp1 = decoder_input @ W_dec + b_dec     # (B, D_obs)
    
    return pred_obs_tp1