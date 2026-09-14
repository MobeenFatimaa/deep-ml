def relu(z: float) -> float:
    """Computes the Rectified Linear Unit (ReLU) activation function."""
    return max(0.0, float(z))


# Example usage
print(relu(0))   # Output: 0.0
print(relu(1))   # Output: 1.0
print(relu(-1))  # Output: 0.0