import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """
    Implements a simple residual block with a shortcut connection.
    
    Args:
        x: 1D input array
        w1: Weight matrix for the first layer
        w2: Weight matrix for the second layer
        
    Returns:
        1D numpy array after residual connection and final ReLU
    """
    # 1. First layer: matrix multiplication followed by ReLU activation
    h1 = np.maximum(0, w1 @ x)
    
    # 2. Second layer: matrix multiplication
    h2 = w2 @ h1
    
    # 3. Shortcut connection: add the original input x
    added = h2 + x
    
    # 4. Final ReLU activation
    out = np.maximum(0, added)
    
    return out