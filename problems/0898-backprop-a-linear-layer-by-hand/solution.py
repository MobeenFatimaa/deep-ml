import torch

def linear_backward(grad_output, x, W):
    """
    Computes the gradients for a linear layer y = x @ W.T + b.
    
    Parameters:
    - grad_output (dL/dy): Tensor of shape (N, out_features)
    - x: Forward input Tensor of shape (N, in_features)
    - W: Weight Tensor of shape (out_features, in_features)
    
    Returns:
    - grad_input (dL/dx): Tensor of shape (N, in_features)
    - grad_W (dL/dW): Tensor of shape (out_features, in_features)
    - grad_b (dL/db): Tensor of shape (out_features,)
    """
    # dL/dx = dL/dy @ W  -> (N, out) @ (out, in) = (N, in)
    grad_input = torch.matmul(grad_output, W)
    
    # dL/dW = dL/dy.T @ x -> (out, N) @ (N, in) = (out, in)
    grad_W = torch.matmul(grad_output.T, x)
    
    # dL/db = sum(dL/dy, dim=0) -> reduce across batch dimension N = (out,)
    grad_b = torch.sum(grad_output, dim=0)
    
    return grad_input, grad_W, grad_b