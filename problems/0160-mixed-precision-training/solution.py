import numpy as np

class MixedPrecision:
    def __init__(self, loss_scale=1024.0):
        """
        Initialize with loss scaling factor.
        """
        self.loss_scale = float(loss_scale)
    
    def forward(self, weights, inputs, targets):
        """
        Perform the forward pass of a linear model in float16, 
        compute MSE loss, scale it, and return as a Python float.
        """
        # Cast weights, inputs, and targets to float16
        w_f16 = weights.astype(np.float16)
        i_f16 = inputs.astype(np.float16)
        t_f16 = targets.astype(np.float16)
        
        # Compute predictions using matrix multiplication in float16
        predictions = i_f16 @ w_f16
        
        # Compute Mean Squared Error (MSE) loss in float16
        mse_loss = np.mean((predictions - t_f16) ** 2)
        
        # Convert loss to a standard Python float and multiply by loss_scale
        scaled_loss = float(mse_loss) * self.loss_scale
        return scaled_loss
    
    def backward(self, gradients):
        """
        Unscale gradients, check for overflow (NaN or Inf), 
        and return as float32 (or zeros if overflow detected).
        """
        grads_arr = np.asarray(gradients)
        
        # Check for overflow (any NaN or Infinite values)
        if not np.isfinite(grads_arr).all():
            return np.zeros_like(grads_arr, dtype=np.float32)
        
        # Unscale gradients and cast to float32
        return (grads_arr / self.loss_scale).astype(np.float32)