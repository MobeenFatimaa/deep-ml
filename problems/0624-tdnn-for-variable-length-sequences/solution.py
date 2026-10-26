import numpy as np

def tdnn_forward(sequences, layer_configs):
    """
    Forward pass through a multi-layer Time-Delay Neural Network.

    Args:
        sequences: list of numpy arrays, each shape (T_i, D_input)
        layer_configs: list of dicts with keys:
            'weights': np.ndarray of shape (len(offsets)*D_in, D_out)
            'bias': np.ndarray of shape (D_out,)
            'offsets': list of int time offsets
            'activation': 'relu' or 'none'

    Returns:
        List of nested Python lists, one per input sequence,
        with values rounded to 4 decimal places.
    """
    results = []
    
    for seq in sequences:
        current_x = seq
        valid = True
        
        for config in layer_configs:
            weights = config['weights']
            bias = config['bias']
            offsets = config['offsets']
            activation = config['activation']
            
            T, D_in = current_x.shape
            min_o = min(offsets)
            max_o = max(offsets)
            
            start_t = max(0, -min_o)
            end_t = T - max_o
            
            if start_t >= end_t:
                valid = False
                break
                
            layer_outputs = []
            for t in range(start_t, end_t):
                concat_vec = np.concatenate([current_x[t + o] for o in offsets])
                out = np.dot(concat_vec, weights) + bias
                if activation == 'relu':
                    out = np.maximum(0, out)
                layer_outputs.append(out)
                
            current_x = np.array(layer_outputs)
            
        if not valid or len(current_x) == 0:
            results.append([])
        else:
            rounded = np.round(current_x, 4)
            results.append(rounded.tolist())
            
    return results