import numpy as np

def conv3d_forward_pass(
    input_volume: np.ndarray,
    kernel: np.ndarray,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Perform 3D convolution forward pass for a single kernel filter.
    
    Args:
        input_volume: Shape (C, D, H, W)
        kernel: Shape (C, kD, kH, kW)
        stride: (stride_d, stride_h, stride_w)
        padding: (pad_d, pad_h, pad_w)
    
    Returns:
        Output volume: Shape (1, D_out, H_out, W_out)
    """
    C, D, H, W = input_volume.shape
    kC, kD, kH, kW = kernel.shape
    
    assert C == kC, f"Channel mismatch: input has {C} channels, kernel has {kC}"
    
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    
    # 1. Apply zero-padding along Depth, Height, and Width
    padded_input = np.pad(
        input_volume,
        pad_width=(
            (0, 0),             # Channels (no padding)
            (pad_d, pad_d),     # Depth
            (pad_h, pad_h),     # Height
            (pad_w, pad_w)      # Width
        ),
        mode='constant',
        constant_values=0
    )
    
    # 2. Calculate spatial output dimensions
    D_out = (D + 2 * pad_d - kD) // stride_d + 1
    H_out = (H + 2 * pad_h - kH) // stride_h + 1
    W_out = (W + 2 * pad_w - kW) // stride_w + 1
    
    # 3. Initialize output feature map (Shape: (1, D_out, H_out, W_out))
    out = np.zeros((1, D_out, H_out, W_out), dtype=input_volume.dtype)
    
    # 4. Sliding window convolution over spatial/temporal dimensions
    for d in range(D_out):
        d_start = d * stride_d
        d_end = d_start + kD
        
        for h in range(H_out):
            h_start = h * stride_h
            h_end = h_start + kH
            
            for w in range(W_out):
                w_start = w * stride_w
                w_end = w_start + kW
                
                # Extract 3D patch across all channels: shape (C, kD, kH, kW)
                patch = padded_input[:, d_start:d_end, h_start:h_end, w_start:w_end]
                
                # Compute sum of element-wise product over channels, depth, height, and width
                out[0, d, h, w] = np.sum(patch * kernel)
                
    return out