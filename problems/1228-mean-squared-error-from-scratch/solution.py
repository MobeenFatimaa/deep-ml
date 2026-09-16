import torch


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Computes Mean Squared Error loss between predictions and targets.

    MSE = mean((pred - target)^2)
    """
    diff = pred - target
    squared_diff = diff**2
    mean_loss = torch.mean(squared_diff)
    return mean_loss.item()


# Example usage:
if __name__ == "__main__":
    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([3.0, 4.0])
    print(mse(pred, target))  # Output: 4.0