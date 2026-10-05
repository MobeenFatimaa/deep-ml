import numpy as np


def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
    """Solve the system Ax = b using the Conjugate Gradient method.

    :param A: Symmetric positive-definite matrix
    :param b: Right-hand side vector
    :param n: Maximum number of iterations
    :param x0: Initial guess for solution (default is zero vector)
    :param tol: Convergence tolerance
    :return: Solution vector x
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    # Step 1: Initialize solution vector x
    if x0 is None:
        x = np.zeros_like(b, dtype=float)
    else:
        x = np.array(x0, dtype=float).copy()

    # Step 2: Compute initial residual r_0 = b - A x_0
    r = b - np.dot(A, x)

    # Step 3: Initialize search direction p_0 = r_0
    p = r.copy()

    rs_old = np.dot(r, r)

    # Check if initial guess already satisfies tolerance
    if np.sqrt(rs_old) < tol:
        return x

    # Step 4: Iterative optimization loop
    for _ in range(n):
        Ap = np.dot(A, p)

        # Step size alpha_k
        alpha = rs_old / np.dot(p, Ap)

        # Update solution x_{k+1}
        x = x + alpha * p

        # Update residual r_{k+1}
        r = r - alpha * Ap

        rs_new = np.dot(r, r)

        # Check convergence criterion
        if np.sqrt(rs_new) < tol:
            break

        # Compute Gram-Schmidt conjugate direction factor beta_k
        beta = rs_new / rs_old

        # Update search direction p_{k+1}
        p = r + beta * p

        rs_old = rs_new

    return x