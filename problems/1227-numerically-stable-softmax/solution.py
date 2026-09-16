import torch


def softmax(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Computes the softmax along a given dimension in a numerically stable way.

    softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    """
    # Find max along dim and keep dimensions for proper broadcasting
    max_val, _ = torch.max(t, dim=dim, keepdim=True)

    # Subtract max for numerical stability (prevents overflow in exp)
    shifted_exp = torch.exp(t - max_val)

    # Sum along dim with keepdim=True for broadcasting during division
    sum_exp = torch.sum(shifted_exp, dim=dim, keepdim=True)

    return shifted_exp / sum_exp


# Example usage:
if __name__ == "__main__":
    t = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    print(softmax(t, dim=1))