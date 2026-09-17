import math

def single_neuron_model(features, labels, weights, bias):
    """
    Simulates a single neuron with a sigmoid activation function.
    
    Args:
        features (list[list[float]]): List of feature vectors (N samples x D features).
        labels (list[int]): List of true binary labels (0 or 1).
        weights (list[float]): List of weights corresponding to each feature.
        bias (float): Bias term.
        
    Returns:
        tuple: (predicted_probabilities, mean_squared_error)
            - predicted_probabilities (list[float]): Probabilities rounded to 4 decimal places.
            - mean_squared_error (float): MSE rounded to 4 decimal places.
    """
    probabilities = []
    total_squared_error = 0.0
    
    for x, y in zip(features, labels):
        # Calculate linear combination (z = w · x + b)
        z = sum(w * x_i for w, x_i in zip(weights, x)) + bias
        
        # Apply Sigmoid activation: 1 / (1 + e^(-z))
        p = 1.0 / (1.0 + math.exp(-z))
        probabilities.append(round(p, 4))
        
        # Accumulate Squared Error: (p - y)^2
        total_squared_error += (p - y) ** 2
    
    # Calculate Mean Squared Error (MSE)
    mse = round(total_squared_error / len(labels), 4)
    
    return probabilities, mse


# Example usage:
if __name__ == "__main__":
    features = [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]]
    labels = [0, 1, 0]
    weights = [0.7, -0.4]
    bias = -0.1

    probs, mse = single_neuron_model(features, labels, weights, bias)
    print(f"Output: ({probs}, {mse})")