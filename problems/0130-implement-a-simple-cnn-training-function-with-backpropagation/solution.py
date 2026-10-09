import numpy as np

def train_simple_cnn_with_backprop(X, y, epochs, learning_rate, kernel_size=3, num_filters=1):
    '''
    Trains a simple CNN with one convolutional layer, ReLU activation, flattening, and a dense layer with softmax output using backpropagation.

    Assumes X has shape (n_samples, height, width) for grayscale images and y is one-hot encoded with shape (n_samples, num_classes).

    Parameters:
    X : np.ndarray, input data
    y : np.ndarray, one-hot encoded labels
    epochs : int, number of training epochs
    learning_rate : float, learning rate for weight updates
    kernel_size : int, size of the square convolutional kernel
    num_filters : int, number of filters in the convolutional layer

    Returns:
    W_conv, b_conv, W_dense, b_dense : Trained weights and biases for the convolutional and dense layers
    '''
    n_samples, height, width = X.shape
    num_classes = y.shape[1]

    # Initialize weights and biases
    W_conv = np.random.randn(kernel_size, kernel_size, num_filters) * 0.01
    b_conv = np.zeros(num_filters)
    output_height = height - kernel_size + 1
    output_width = width - kernel_size + 1
    flattened_size = output_height * output_width * num_filters
    W_dense = np.random.randn(flattened_size, num_classes) * 0.01
    b_dense = np.zeros(num_classes)

    for epoch in range(epochs):
        for i in range(n_samples):
            x_i = X[i]  # Shape: (height, width)
            y_i = y[i]  # Shape: (num_classes,)

            # -----------------------------------------------------------------
            # 1. FORWARD PASS
            # -----------------------------------------------------------------
            # Convolution (valid padding, stride=1)
            conv_out = np.zeros((output_height, output_width, num_filters))
            for f in range(num_filters):
                for h in range(output_height):
                    for w in range(output_width):
                        patch = x_i[h:h+kernel_size, w:w+kernel_size]
                        conv_out[h, w, f] = np.sum(patch * W_conv[:, :, f]) + b_conv[f]

            # ReLU Activation
            relu_out = np.maximum(0, conv_out)

            # Flattening
            flattened_out = relu_out.flatten()

            # Dense Layer
            dense_logits = np.dot(flattened_out, W_dense) + b_dense

            # Softmax Output
            exp_logits = np.exp(dense_logits - np.max(dense_logits))
            probs = exp_logits / np.sum(exp_logits)

            # -----------------------------------------------------------------
            # 2. BACKWARD PASS (Gradients computation)
            # -----------------------------------------------------------------
            # Gradient of Categorical Cross-Entropy Loss with Softmax: dL/d(logits) = probs - y
            d_logits = probs - y_i

            # Gradients for Dense Layer weights and biases
            dW_dense = np.outer(flattened_out, d_logits)
            db_dense = d_logits

            # Gradient wrt Dense input (Flattened output)
            d_flattened = np.dot(W_dense, d_logits)

            # Unflatten gradient back to feature map dimensions
            d_relu = d_flattened.reshape(output_height, output_width, num_filters)

            # Gradient wrt ReLU activation
            d_conv = d_relu * (conv_out > 0)

            # Gradients for Convolutional Layer weights and biases
            dW_conv = np.zeros_like(W_conv)
            db_conv = np.zeros_like(b_conv)

            for f in range(num_filters):
                db_conv[f] = np.sum(d_conv[:, :, f])
                for h in range(output_height):
                    for w in range(output_width):
                        patch = x_i[h:h+kernel_size, w:w+kernel_size]
                        dW_conv[:, :, f] += patch * d_conv[h, w, f]

            # -----------------------------------------------------------------
            # 3. WEIGHT & BIAS UPDATES (Stochastic Gradient Descent)
            # -----------------------------------------------------------------
            W_dense -= learning_rate * dW_dense
            b_dense -= learning_rate * db_dense
            W_conv -= learning_rate * dW_conv
            b_conv -= learning_rate * db_conv

    return W_conv, b_conv, W_dense, b_dense