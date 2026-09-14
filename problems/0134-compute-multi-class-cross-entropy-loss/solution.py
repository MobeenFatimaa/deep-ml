import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, 
                               true_labels: np.ndarray, 
                               eps: float = 1e-15) -> float:
    """Computes average multi-class cross-entropy loss for a batch of predictions."""
    y_pred = np.array(predicted_probs, dtype=np.float64)
    y_true = np.array(true_labels, dtype=np.float64)
    
    # Clip probabilities to avoid log(0) numerical instability
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # Compute cross-entropy loss
    sample_losses = -np.sum(y_true * np.log(y_pred), axis=-1)
    return float(np.mean(sample_losses))


# Test Case verification
if __name__ == "__main__":
    pred = np.array([[1, 0, 0], [0, 1, 0]])
    true = np.array([[1, 0, 0], [0, 1, 0]])
    print(round(compute_cross_entropy_loss(pred, true), 4))
    # Output: 0.0