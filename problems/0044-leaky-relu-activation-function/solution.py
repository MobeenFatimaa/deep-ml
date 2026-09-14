def leaky_relu(z: float, alpha: float = 0.01) -> float:
    """Computes the Leaky Rectified Linear Unit (Leaky ReLU) activation function."""
    return z if z > 0 else alpha * z


# Example usage
print(leaky_relu(0))                # Output: 0
print(leaky_relu(1))                # Output: 1
print(leaky_relu(-1))               # Output: -0.01
print(leaky_relu(-2, alpha=0.1))    # Output: -0.2