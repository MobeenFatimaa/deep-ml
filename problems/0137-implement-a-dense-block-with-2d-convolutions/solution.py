import numpy as np

def dense_net_block(input_data, num_layers, growth_rate, kernels, kernel_size=(3, 3)):
    """
    Performs the forward pass of a DenseNet dense block.

    Args:
        input_data: Input tensor of shape (N, H, W, C0).
        num_layers: Int, number of convolutional layers in the block.
        growth_rate: Int, number of feature maps added per layer (out_channels).
        kernels: List of filter weights, where kernels[l] has shape (kh, kw, C_in, growth_rate).
        kernel_size: Tuple (kh, kw), spatial dimensions of the kernel. Default (3, 3).

    Returns:
        Tensor of shape (N, H, W, C0 + num_layers * growth_rate).
    """
    current_features = np.array(input_data, copy=True)
    kh, kw = kernel_size
    pad_h = kh // 2
    pad_w = kw // 2

    for l in range(num_layers):
        N, H, W, C_in = current_features.shape
        kernel = kernels[l]

        # Validate input channel alignment
        if kernel.shape[2] != C_in:
            raise ValueError(
                f"Layer {l} kernel expects {kernel.shape[2]} input channels, "
                f"but feature map has {C_in} channels."
            )

        # 1. Apply ReLU activation
        activated_features = np.maximum(0, current_features)

        # 2. Symmetric zero-padding along spatial dimensions (H, W)
        padded_features = np.pad(
            activated_features,
            pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode='constant',
            constant_values=0
        )

        # 3. Perform 2D Convolution (NHWC layout)
        conv_out = np.zeros((N, H, W, growth_rate), dtype=current_features.dtype)

        for n in range(N):
            for h in range(H):
                for w in range(W):
                    # Extract patch matching spatial kernel bounds
                    patch = padded_features[n, h:h + kh, w:w + kw, :]  # Shape: (kh, kw, C_in)
                    # Contract spatial (kh, kw) and input channel (C_in) dimensions
                    conv_out[n, h, w, :] = np.tensordot(patch, kernel, axes=([0, 1, 2], [0, 1, 2]))

        # 4. Concatenate new features along the channel axis (axis=-1)
        current_features = np.concatenate([current_features, conv_out], axis=-1)

    return current_features