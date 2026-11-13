import numpy as np

def newtonschulz5(G: np.ndarray, steps=5, eps=1e-7) -> np.ndarray:
    """
    Apply the Newton-Schulz (quintic) iteration for 5 steps to matrix G.
    Args:
        G: 2D NumPy array
        steps: Number of iteration steps (default 5)
        eps: Small constant for stability
    Returns:
        Matrix after Newton-Schulz iteration
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    
    # If G has more rows than columns, run the iteration on G^T and transpose back
    transposed = G.shape[0] > G.shape[1]
    X = G.T if transposed else G.copy()
    
    # Initial normalization by Frobenius norm + eps
    X = X / (np.linalg.norm(X) + eps)
    
    for _ in range(steps):
        A = X @ X.T
        # Quintic Newton-Schulz iteration update
        X = a * X + (b * A + c * (A @ A)) @ X
        
    if transposed:
        X = X.T
        
    return X

def muon_update(theta: np.ndarray, grad: np.ndarray, B_prev: np.ndarray, mu: float, lr: float) -> tuple:
    """
    Performs one Muon optimizer update. Returns the updated parameter, new B, and the preconditioned update.
    Args:
        theta: Parameter matrix (2D NumPy array)
        grad: Gradient matrix (same shape)
        B_prev: Previous B matrix (momentum)
        mu: Momentum factor (0 <= mu < 1)
        lr: Learning rate (step size)
    Returns:
        theta_new: Updated parameter matrix
        B_new: Updated B matrix
        O: Preconditioned update
    """
    # Step 1: Update momentum buffer B_t = mu * B_{t-1} + grad
    B_new = mu * B_prev + grad
    
    # Step 2: Compute preconditioned orthogonalized update via Newton-Schulz
    O = newtonschulz5(B_new, steps=5, eps=1e-7)
    
    # Step 3: Update parameters theta_t = theta_{t-1} - lr * O
    theta_new = theta - lr * O
    
    return theta_new, B_new, O