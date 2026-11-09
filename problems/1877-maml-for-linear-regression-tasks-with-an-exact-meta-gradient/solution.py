import numpy as np

def adapt(xi, X, y, inner_lr, inner_steps):
    """
    Run inner_steps steps of gradient descent on L(.; X, y) from xi with step size inner_lr.
    Returns the adapted parameters theta (1D NumPy array).
    """
    n = X.shape[0]
    theta = np.array(xi, dtype=float)
    if inner_steps == 0 or n == 0:
        return theta
    
    for _ in range(inner_steps):
        # Gradient of L(theta; X, y) = (1 / (2n)) ||X theta - y||^2
        # grad = (1 / n) * X^T @ (X @ theta - y)
        grad = (1.0 / n) * X.T @ (X @ theta - y)
        theta = theta - inner_lr * grad
        
    return theta

def meta_loss_and_grad(xi, tasks, inner_lr, inner_steps, first_order=False):
    """
    Compute the meta-loss and its exact (or first-order) gradient with respect to xi.
    Returns (loss, grad) where loss is a Python float and grad is a 1D NumPy array.
    """
    J = len(tasks)
    total_loss = 0.0
    total_grad = np.zeros_like(xi, dtype=float)
    
    for X_tr, y_tr, X_va, y_va in tasks:
        n_tr = X_tr.shape[0]
        n_va = X_va.shape[0]
        
        # 1. Forward pass: adapt parameters from xi
        theta = np.array(xi, dtype=float)
        
        if not first_order and inner_steps > 0:
            # Transition matrix A for linear regression inner updates:
            # theta_{t+1} = theta_t - inner_lr * (1 / n_tr) * X_tr^T (X_tr theta_t - y_tr)
            #             = (I - inner_lr / n_tr * X_tr^T X_tr) theta_t + constant
            A = np.eye(len(xi)) - inner_lr * (1.0 / n_tr) * (X_tr.T @ X_tr)
        
        for _ in range(inner_steps):
            grad_train = (1.0 / n_tr) * X_tr.T @ (X_tr @ theta - y_tr)
            theta = theta - inner_lr * grad_train
            
        theta_hat = theta
        
        # 2. Validation loss for the task
        residuals = X_va @ theta_hat - y_va
        task_loss = (0.5 / n_va) * np.sum(residuals ** 2)
        total_loss += task_loss
        
        # 3. Gradient of validation loss w.r.t. theta_hat
        grad_theta_va = (1.0 / n_va) * X_va.T @ residuals
        
        # 4. Backward pass / Chain rule for meta-gradient
        if first_order or inner_steps == 0:
            total_grad += grad_theta_va
        else:
            # Exact gradient accounts for derivative through inner steps: d(theta_hat) / d(xi) = A^S
            v = grad_theta_va
            for _ in range(inner_steps):
                v = v @ A
            total_grad += v
            
    return float(total_loss / J), total_grad / J

def maml_step(xi, tasks, inner_lr, outer_lr, inner_steps, first_order=False):
    """
    Perform one outer gradient-descent step on the meta-loss and return the updated xi.
    """
    _, grad = meta_loss_and_grad(xi, tasks, inner_lr, inner_steps, first_order=first_order)
    return xi - outer_lr * grad