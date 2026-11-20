import numpy as np


def sgtm_step(
    params: np.ndarray,
    grad: np.ndarray,
    forget_mask: np.ndarray,
    lr: float = 0.1,
    batch_type: str = "forget",
) -> np.ndarray:
    """Perform one SGTM update step on a 1D parameter vector.

    SGTM splits parameters into 'forget' and 'retain' groups using a binary mask.
    Depending on the batch type, only some parameters are allowed to update.

    Rules:
    - 'forget' batch: only forget parameters update (retain params get zeroed gradients).
    - 'retain' batch: only retain parameters update (forget params get zeroed gradients).
    - 'unlabeled' batch: all parameters update normally.

    Any nonzero entry in forget_mask should be treated as 1 (forget), and zero
    means retain.

    Args:
        params: Current parameters, shape (d,)
        grad: Gradient for this batch, shape (d,)
        forget_mask: Mask for forget parameters, shape (d,)
        lr: Learning rate
        batch_type: One of {'forget', 'retain', 'unlabeled'}

    Returns:
        new_params: Updated parameters
    """
    params = np.asarray(params, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    is_forget = np.asarray(forget_mask) != 0

    if batch_type == "forget":
        grad_mask = is_forget
    elif batch_type == "retain":
        grad_mask = ~is_forget
    elif batch_type == "unlabeled":
        grad_mask = np.ones_like(is_forget, dtype=bool)
    else:
        raise ValueError(
            f"Invalid batch_type '{batch_type}'. Expected one of {{'forget', 'retain', 'unlabeled'}}."
        )

    masked_grad = np.where(grad_mask, grad, 0.0)
    new_params = params - float(lr) * masked_grad
    return new_params