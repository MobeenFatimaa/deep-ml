import numpy as np

def smooth_labels(y_true, num_classes, epsilon):
    """
    Create smoothed one-hot target vectors.

    Args:
        y_true: Iterable[int] of shape (N,) with values in [0, K-1]
        num_classes: int, total number of classes (K)
        epsilon: float in [0, 1]

    Returns:
        np.ndarray of shape (N, K) with smoothed probabilities.
    """
    if not (0.0 <= epsilon <= 1.0):
        raise ValueError("Epsilon must be between 0 and 1.")
        
    y_true = np.asarray(y_true)
    N = y_true.shape[0]
    
    # Initialize array with uniform distribution (epsilon / K)
    smooth = np.full((N, num_classes), epsilon / num_classes, dtype=np.float64)
    
    # Add (1 - epsilon) to the true class indices
    smooth[np.arange(N), y_true] += (1.0 - epsilon)
    
    return smooth


def label_smoothing_cross_entropy(logits, y_true, num_classes, epsilon=0.1, round_decimals=None):
    """
    Compute mean cross-entropy between logits and smoothed targets using stable log-softmax.

    Args:
        logits: Array-like of shape (N, K), model output scores.
        y_true: Array-like of shape (N,), integer class indices.
        num_classes: int, number of classes (K).
        epsilon: float in [0, 1].
        round_decimals: int | None, round the loss to this many decimals if given.

    Returns:
        float: Mean cross-entropy loss.
    """
    logits = np.asarray(logits, dtype=np.float64)
    y_true = np.asarray(y_true)

    # Validate dimensions
    if logits.ndim != 2 or logits.shape[1] != num_classes:
        raise ValueError("Logits must be 2-D and its second dimension must equal num_classes.")

    # 1. Numerically stable log-softmax (subtracting row max)
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted_logits), axis=1, keepdims=True))
    log_softmax = shifted_logits - log_sum_exp

    # 2. Get smoothed target distributions
    targets = smooth_labels(y_true, num_classes, epsilon)

    # 3. Compute cross-entropy loss per sample: - sum(t_k * log_softmax_k)
    sample_losses = -np.sum(targets * log_softmax, axis=1)

    # 4. Compute mean loss across all samples
    mean_loss = float(np.mean(sample_losses))

    if round_decimals is not None:
        mean_loss = round(mean_loss, round_decimals)

    return mean_loss