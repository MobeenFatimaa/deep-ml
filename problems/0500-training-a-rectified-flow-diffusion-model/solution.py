import numpy as np

def rectified_flow_step(
    X_1: np.ndarray,
    X_0: np.ndarray,
    t: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    lr: float
) -> dict:
    """
    Perform one training step for a rectified flow model.

    Args:
        X_1: Data samples, shape (N, D)
        X_0: Noise samples, shape (N, D)
        t: Timesteps in [0, 1], shape (N, 1)
        W: Velocity network weights, shape (D+1, D)
        b: Velocity network bias, shape (1, D)
        lr: Learning rate for gradient descent

    Returns:
        Dictionary with keys:
            'x_t': Interpolated samples, shape (N, D)
            'v_target': Target velocities, shape (N, D)
            'v_pred': Predicted velocities, shape (N, D)
            'loss': Scalar MSE loss (float)
            'W_new': Updated weight matrix, shape (D+1, D)
            'b_new': Updated bias vector, shape (1, D)
    """
    N, D = X_1.shape

    # 1. Interpolation: x_t = (1 - t) * X_0 + t * X_1
    # Since t has shape (N, 1), it broadcasts correctly across D features.
    x_t = (1.0 - t) * X_0 + t * X_1

    # 2. Target computation: velocity pointing from noise (t=0) to data (t=1)
    v_target = X_1 - X_0

    # 3. Forward pass: concatenate x_t and t along the feature axis -> shape (N, D+1)
    X_in = np.concatenate([x_t, t], axis=1)
    v_pred = X_in @ W + b

    # 4. Loss: Mean Squared Error over all batch elements and dimensions
    diff = v_pred - v_target
    loss = float(np.mean(diff ** 2))

    # 5. Backward pass: compute analytical gradients
    # dL / dv_pred = (2 / (N * D)) * diff
    num_elements = N * D
    grad_v = (2.0 / num_elements) * diff

    # Gradients w.r.t weights and bias
    grad_W = X_in.T @ grad_v
    grad_b = np.sum(grad_v, axis=0, keepdims=True)

    # 6. Update parameters using gradient descent
    W_new = W - lr * grad_W
    b_new = b - lr * grad_b

    return {
        'x_t': x_t,
        'v_target': v_target,
        'v_pred': v_pred,
        'loss': loss,
        'W_new': W_new,
        'b_new': b_new
    }