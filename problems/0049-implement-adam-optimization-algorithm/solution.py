import numpy as np


def adam_optimizer(
    f,
    grad,
    x0,
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    num_iterations=10,
):
    # Note: `f` is accepted only for interface parity with the PyTorch/Tinygrad
    # variants, which derive gradients from it via autograd. This version uses
    # `grad` only — the objective value `f` is never evaluated.

    x = np.array(x0, dtype=float)

    # Initialize 1st moment vector (mean of gradients) and 2nd moment vector (uncentered variance of gradients)
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    for t in range(1, num_iterations + 1):
        # Compute gradient at current parameters
        g = grad(x)

        # Update biased 1st moment estimate
        m = beta1 * m + (1 - beta1) * g

        # Update biased 2nd raw moment estimate
        v = beta2 * v + (1 - beta2) * (g**2)

        # Compute bias-corrected 1st moment estimate
        m_hat = m / (1 - beta1**t)

        # Compute bias-corrected 2nd raw moment estimate
        v_hat = v / (1 - beta2**t)

        # Update parameters
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    return x