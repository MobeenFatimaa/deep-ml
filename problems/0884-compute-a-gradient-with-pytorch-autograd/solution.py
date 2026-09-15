import torch

def grad_of_quadratic(x_value: float) -> float:
    """
    Computes f'(x) for f(x) = x^2 + 3x + 2 at x_value using PyTorch autograd.
    
    Args:
        x_value: The scalar point at which to evaluate the gradient.
        
    Returns:
        The gradient as a Python float.
    """
    # 1. Create a leaf tensor with requires_grad=True
    x = torch.tensor(float(x_value), requires_grad=True)
    
    # 2. Compute the forward pass
    y = x ** 2 + 3 * x + 2
    
    # 3. Trigger backpropagation
    y.backward()
    
    # 4. Extract and return the gradient value as a python float
    return x.grad.item()