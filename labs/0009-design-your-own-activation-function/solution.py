import numpy as np

def activation(x):
    '''
    Apply SiLU (Sigmoid Linear Unit / Swish) element-wise.
    
    f(x) = x / (1 + exp(-x))
    '''
    return x / (1.0 + np.exp(-np.clip(x, -500, 500))) # Clipped to prevent numerical overflow