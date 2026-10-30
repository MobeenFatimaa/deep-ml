import numpy as np


def standardize(X_train, X_test):
    """
    Center and scale X_train and X_test using the training set means and standard deviations (ddof=0).
    
    Args:
        X_train: Training data array of shape (N_train, D)
        X_test: Test data array of shape (N_test, D)
        
    Returns:
        tuple: (Z_train, Z_test) standardized arrays
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0, ddof=0)
    
    # Handle zero standard deviations to avoid division by zero
    std[std == 0.0] = 1.0
    
    Z_train = (X_train - mean) / std
    Z_test = (X_test - mean) / std
    
    return Z_train, Z_test


def pcr_fit_predict(X_train, y_train, X_test, m):
    """
    Perform Principal Components Regression (PCR) using m components.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    
    # 1. Standardize data
    Z_train, Z_test = standardize(X_train, X_test)
    
    # 2. SVD of Z_train to get principal directions
    # U, S, Vt = np.linalg.svd(Z_train, full_matrices=False)
    # The rows of Vt are the right-singular vectors (principal directions)
    _, _, Vt = np.linalg.svd(Z_train, full_matrices=False)
    V_m = Vt[:m].T  # Shape: (D, m)
    
    # 3. Project train and test onto the first m components (scores)
    T_train = Z_train @ V_m  # Shape: (N_train, m)
    T_test = Z_test @ V_m    # Shape: (N_test, m)
    
    # 4. Least squares regression of y_train on training scores with an intercept
    y_mean = np.mean(y_train)
    y_centered = y_train - y_mean
    
    # Add a column of ones to training scores for the intercept
    T_train_design = np.column_stack([np.ones(len(T_train)), T_train])
    T_test_design = np.column_stack([np.ones(len(T_test)), T_test])
    
    # Solve normal equations or least squares: beta = (T^T T)^(-1) T^T y_centered
    # Note: Using lstsq is robust against collinearity
    beta, _, _, _ = np.linalg.lstsq(T_train_design, y_centered, rcond=None)
    
    # 5. Predict on test set
    predictions = T_test_design @ beta + y_mean
    
    return predictions.flatten()


def pls_fit_predict(X_train, y_train, X_test, m):
    """
    Perform Partial Least Squares (PLS1) using m NIPALS rounds.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    
    # 1. Standardize X and center y by its training mean
    Z_train, Z_test = standardize(X_train, X_test)
    y_mean = np.mean(y_train)
    yk = y_train - y_mean
    
    Xk = Z_train.copy()
    
    N, D = Z_train.shape
    W = np.zeros((D, m))
    P = np.zeros((D, m))
    q = np.zeros(m)
    
    for k in range(m):
        # w = Xk^T yk / ||Xk^T yk||
        Xyk = Xk.T @ yk
        norm_Xyk = np.linalg.norm(Xyk)
        if norm_Xyk < 1e-12:
            w = np.zeros_like(Xyk)
        else:
            w = Xyk / norm_Xyk
        
        # t = Xk w
        t = Xk @ w
        
        # p = Xk^T t / (t^T t)
        t_dot_t = t @ t
        if t_dot_t < 1e-12:
            p = np.zeros(D)
        else:
            p = (Xk.T @ t) / t_dot_t
            
        # q = t^T yk / (t^T t)
        if t_dot_t < 1e-12:
            q_k = 0.0
        else:
            q_k = (t @ yk) / t_dot_t
            
        W[:, k] = w
        P[:, k] = p
        q[k] = q_k
        
        # Deflate Xk and yk
        Xk = Xk - np.outer(t, p)
        yk = yk - q_k * t
        
    # 2. Compute regression coefficients B = W (P^T W)^-1 q
    PtW = P.T @ W
    # Use pseudoinverse to handle potential rank deficiencies safely
    B = W @ np.linalg.pinv(PtW) @ q
    
    # 3. Predict on test set: Z_test @ B + mean(y_train)
    predictions = Z_test @ B + y_mean
    
    return predictions.flatten()