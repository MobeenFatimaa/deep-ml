import numpy as np


def simple_conv2d(
    input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int
) -> np.ndarray:
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    # Apply zero-padding to the input matrix if padding > 0
    if padding > 0:
        padded_input = np.pad(
            input_matrix,
            pad_width=((padding, padding), (padding, padding)),
            mode="constant",
            constant_values=0,
        )
    else:
        padded_input = input_matrix

    # Calculate output dimensions
    padded_height, padded_width = padded_input.shape
    output_height = (padded_height - kernel_height) // stride + 1
    output_width = (padded_width - kernel_width) // stride + 1

    output_matrix = np.zeros((output_height, output_width), dtype=float)

    # Perform 2D convolution (element-wise multiplication and sum)
    for i in range(output_height):
        for j in range(output_width):
            row_start = i * stride
            row_end = row_start + kernel_height
            col_start = j * stride
            col_end = col_start + kernel_width

            # Extract receptive field patch and compute dot product with kernel
            patch = padded_input[row_start:row_end, col_start:col_end]
            output_matrix[i, j] = np.sum(patch * kernel)

    return output_matrix