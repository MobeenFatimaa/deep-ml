import torch


def bce_with_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean BCE-with-logits loss, numerically stable, rounded to 4 decimals.

    Args:
        logits (torch.Tensor): 1-D raw logits.
        targets (torch.Tensor): 1-D binary targets in {0, 1}, same shape.

    Returns:
        float: mean loss rounded to 4 decimal places.
    """
    # Numerically stable per-element BCE loss formula:
    # max(x, 0) - x * y + log(1 + exp(-|x|))
    max_val = torch.clamp(logits, min=0.0)
    loss = max_val - logits * targets + torch.log1p(torch.exp(-torch.abs(logits)))

    return round(torch.mean(loss).item(), 4)