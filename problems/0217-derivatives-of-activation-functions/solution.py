import math

def activation_derivatives(x: float) -> dict[str, float]:
    """Computes the derivatives of Sigmoid, Tanh, and ReLU activation functions at x."""
    # Sigmoid: σ(x) = 1 / (1 + exp(-x))
    # Derivative: σ'(x) = σ(x) * (1 - σ(x))
    sigmoid_val = 1.0 / (1.0 + math.exp(-x))
    d_sigmoid = sigmoid_val * (1.0 - sigmoid_val)
    
    # Tanh: tanh(x)
    # Derivative: tanh'(x) = 1 - tanh(x)^2
    tanh_val = math.tanh(x)
    d_tanh = 1.0 - (tanh_val ** 2)
    
    # ReLU: f(x) = max(0, x)
    # Derivative: f'(x) = 1.0 if x > 0 else 0.0
    d_relu = 1.0 if x > 0 else 0.0
    
    return {
        'sigmoid': d_sigmoid,
        'tanh': d_tanh,
        'relu': d_relu
    }


if __name__ == "__main__":
    result = activation_derivatives(0.0)
    print({k: round(v, 4) for k, v in result.items()})
    # Output: {'sigmoid': 0.25, 'tanh': 1.0, 'relu': 0.0}