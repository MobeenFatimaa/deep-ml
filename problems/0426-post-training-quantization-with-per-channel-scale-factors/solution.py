import numpy as np

def per_channel_quantize(weight: np.ndarray, bits: int = 8) -> tuple:
    """
    Perform symmetric per-channel post-training quantization.

    Args:
        weight: Weight matrix of shape (out_channels, in_features)
        bits: Target bit-width for quantization (default: 8)

    Returns:
        Tuple of (quantized_weights, scale_factors, dequantized_weights)
        - quantized_weights: int array of shape (out_channels, in_features)
        - scale_factors: float array of shape (out_channels,)
        - dequantized_weights: float array of shape (out_channels, in_features)
    """
    weight = np.asarray(weight, dtype=np.float64)
    out_channels = weight.shape[0]
    
    # Calculate quantization integer bounds
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    
    # Find absolute maximum value for each row/channel
    max_abs_per_channel = np.max(np.abs(weight), axis=1)
    
    # Calculate scale factor per channel (handle all-zero channels gracefully)
    scale_factors = np.where(
        max_abs_per_channel == 0.0,
        1.0,
        max_abs_per_channel / qmax
    )
    
    # Expand scales for broadcasting across columns: shape (out_channels, 1)
    scale_expanded = scale_factors[:, np.newaxis]
    
    # Quantize and clip to [qmin, qmax]
    q_float = weight / scale_expanded
    quantized_weights = np.clip(np.round(q_float), qmin, qmax).astype(np.int64)
    
    # Dequantize back to float representation
    dequantized_weights = quantized_weights * scale_expanded
    
    return quantized_weights, scale_factors, dequantized_weights