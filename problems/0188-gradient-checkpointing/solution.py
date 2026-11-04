import numpy as np

def checkpoint_forward(funcs, input_arr):
    """
    Applies a list of functions in sequence to the input array, simulating gradient checkpointing by not storing intermediates.

    Args:
        funcs (list of callables): List of functions to apply in sequence.
        input_arr (np.ndarray): Input numpy array.

    Returns:
        np.ndarray: The output after applying all functions, of type float and same shape as final output.
    """
    current = np.asarray(input_arr, dtype=float)
    
    for f in funcs:
        current = f(current)
        
    return np.asarray(current, dtype=float)