import torch


def relu(t: torch.Tensor) -> torch.Tensor:
    """Computes the Rectified Linear Unit (ReLU) element-wise.

    ReLU(x) = max(0, x)
    """
    return torch.clamp(t, min=0.0)


def leaky_relu(t: torch.Tensor, slope: float = 0.01) -> torch.Tensor:
    """Computes the Leaky ReLU element-wise.

    Leaky_ReLU(x) = x if x > 0 else slope * x
    """
    return torch.where(t > 0, t, t * slope)


# Example usage:
if __name__ == "__main__":
    t = torch.tensor([-2.0, -0.5, 0.0, 1.5])
    print(relu(t))
    print(leaky_relu(t, 0.1))