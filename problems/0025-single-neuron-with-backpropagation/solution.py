import numpy as np

def train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs):
    """
    Trains a single neuron with sigmoid activation using full-batch gradient descent (MSE loss).
    """
    X = np.array(features)
    y = np.array(labels)
    w = np.array(initial_weights, dtype=float)
    b = float(initial_bias)
    N = len(y)
    
    mse_values = []

    for _ in range(epochs):
        # Forward pass: z = Xw + b
        z = np.dot(X, w) + b
        p = 1.0 / (1.0 + np.exp(-z))

        # Record MSE before updating
        mse = np.mean((p - y) ** 2)
        mse_values.append(round(float(mse), 4))

        # Gradient computation: dL/dz = (2/N) * (p - y) * p * (1 - p)
        dL_dz = (2.0 / N) * (p - y) * p * (1.0 - p)
        
        dw = np.dot(X.T, dL_dz)
        db = np.sum(dL_dz)

        # Parameter updates
        w -= learning_rate * dw
        b -= learning_rate * db

    return np.round(w, 4).tolist(), round(float(b), 4), mse_values