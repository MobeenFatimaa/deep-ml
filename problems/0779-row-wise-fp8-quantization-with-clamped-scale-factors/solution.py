import numpy as np

def row_wise_fp8_quantize(X: np.ndarray) -> dict:
    """
    Perform row-wise FP8 quantization with clamped per-row scale factors.

    Args:
        X: 2D numpy array of shape (num_rows, num_cols)

    Returns:
        Dictionary with keys:
            'quantized': quantized 2D numpy array, same shape as X
            'scales':    1D numpy array of per-row scale factors (length num_rows)
    """
    FP8_MAX = 448.0
    SCALE_MAX = 1200.0

    # Step 1: Compute the row's maximum absolute value
    row_max = np.max(np.abs(X), axis=1)

    # Step 2: Compute the scale factor
    scales = row_max / FP8_MAX

    # Step 3: Clamp the scale factor so it does not exceed SCALE_MAX
    scales = np.minimum(scales, SCALE_MAX)

    # Step 4: Handle zeros to avoid division by zero
    scales[scales == 0.0] = 1.0

    # Step 5: Compute the quantized row as row / scale, then clip to [-FP8_MAX, FP8_MAX]
    # Reshape scales to (num_rows, 1) for correct broadcasting
    quantized = np.clip(X / scales[:, np.newaxis], -FP8_MAX, FP8_MAX)

    return {
        'quantized': quantized,
        'scales': scales
    }