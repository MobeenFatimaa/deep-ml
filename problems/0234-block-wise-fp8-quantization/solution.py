import numpy as np

def fp8_block_quantize(
    tensor: np.ndarray,
    block_size: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize a tensor to FP8-E4M3 format using block-wise scaling.
    
    Args:
        tensor: Input tensor of shape (N,) where N is divisible by block_size
        block_size: Number of elements per quantization block
        
    Returns:
        quantized: Quantized values of shape (N,), clipped to [-448, 448]
        scales: Per-block scale factors of shape (N // block_size,)
    """
    FP8_MAX = 448.0
    EPS = 1e-12

    num_blocks = tensor.size // block_size
    # Reshape into blocks of size block_size
    blocks = tensor.reshape(num_blocks, block_size)

    # Compute maximum absolute value for each block
    max_vals = np.max(np.abs(blocks), axis=1)

    # Compute scale factor: sb = (Mb + eps) / FP8_MAX
    scales = (max_vals + EPS) / FP8_MAX

    # Quantize: q = clip(round(x / sb), -448, 448)
    # Broadcast scales across blocks
    quantized_blocks = np.clip(np.round(blocks / scales[:, np.newaxis]), -FP8_MAX, FP8_MAX)

    return quantized_blocks.flatten(), scales


def fp8_block_dequantize(
    quantized: np.ndarray,
    scales: np.ndarray,
    block_size: int = 128
) -> np.ndarray:
    """
    Dequantize FP8-E4M3 values back to full precision.
    
    Args:
        quantized: Quantized values of shape (N,)
        scales: Per-block scale factors of shape (N // block_size,)
        block_size: Number of elements per quantization block
        
    Returns:
        Dequantized tensor of shape (N,)
    """
    num_blocks = quantized.size // block_size
    quantized_blocks = quantized.reshape(num_blocks, block_size)

    # Dequantize: x_hat = q * sb
    dequantized_blocks = quantized_blocks * scales[:, np.newaxis]

    return dequantized_blocks.flatten()