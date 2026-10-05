import numpy as np


def pos_encoding(position: int, d_model: int):
    """Calculates sinusoidal positional encodings for Transformers.

    Args:
        position (int): Sequence length (number of positions).
        d_model (int): Model vector dimensionality.

    Returns:
        np.ndarray or int: NumPy array of shape (position, d_model) with type
        float16, or -1 if position <= 0 or d_model <= 0.
    """
    # Validation check: Return -1 if position is 0 or d_model is <= 0
    if position <= 0 or d_model <= 0:
        return -1

    # Initialize positional encoding matrix
    PE = np.zeros((position, d_model), dtype=np.float32)

    # Positions vector: shape (position, 1)
    pos = np.arange(position)[:, np.newaxis]

    # Dimension indices vector for even indices: shape (d_model // 2,)
    i = np.arange(0, d_model, 2)

    # Compute div_term = 10000^(2i / d_model)
    div_term = np.power(10000.0, i / d_model)

    # Compute sine for even indices: PE(pos, 2i) = sin(pos / 10000^(2i / d_model))
    PE[:, 0::2] = np.sin(pos / div_term)

    # Compute cosine for odd indices: PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    PE[:, 1::2] = np.cos(pos / div_term[: d_model // 2])

    # Convert output to float16 precision
    return PE.astype(np.float16)