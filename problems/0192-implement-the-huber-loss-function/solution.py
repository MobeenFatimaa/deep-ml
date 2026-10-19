def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute the Huber Loss between true and predicted values.

    Args:
        y_true (float | list[float]): Ground truth values
        y_pred (float | list[float]): Predicted values
        delta (float): Transition threshold between MSE and MAE behavior

    Returns:
        float: Average Huber loss
    """
    if isinstance(y_true, (int, float)):
        y_true = [y_true]
    if isinstance(y_pred, (int, float)):
        y_pred = [y_pred]

    total_loss = 0.0
    for yt, yp in zip(y_true, y_pred):
        error = abs(yt - yp)
        if error <= delta:
            total_loss += 0.5 * (error ** 2)
        else:
            total_loss += delta * (error - 0.5 * delta)
            
    return total_loss / len(y_true)