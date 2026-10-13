from tinygrad import Tensor

def dropout(x, p, training):
    if not training or p == 0.0:
        return x
    
    # Generate random uniform noise in [0, 1) with the same shape as x
    noise = Tensor.rand(*x.shape)
    
    # Create a binary mask where elements are kept with probability (1 - p)
    # (noise >= p) is True with probability 1 - p
    mask = (noise >= p).cast(x.dtype)
    
    # Scale surviving elements by 1 / (1 - p) for inverted dropout
    scale = 1.0 / (1.0 - p)
    
    return x * mask * scale