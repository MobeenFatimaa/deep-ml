from typing import Tuple

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    """
    Determine when to stop training early based on validation losses, patience, and min_delta.
    
    Args:
        val_losses (list[float]): List of validation losses per epoch.
        patience (int): Number of epochs to wait for an improvement before stopping.
        min_delta (float): Minimum decrease in loss required to qualify as an improvement.
        
    Returns:
        Tuple[int, int]: (stop_epoch, best_epoch)
    """
    if not val_losses:
        return (0, 0)
        
    best_loss = val_losses[0]
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, len(val_losses)):
        loss = val_losses[epoch]
        
        # Check if the loss improved by more than min_delta
        if loss < best_loss - min_delta:
            best_loss = loss
            best_epoch = epoch
            patience_counter = 0  # Reset counter on improvement
        else:
            patience_counter += 1
            if patience_counter >= patience:
                return (epoch, best_epoch)
                
    # If patience is never exceeded, return the final epoch and the best epoch found
    return (len(val_losses) - 1, best_epoch)