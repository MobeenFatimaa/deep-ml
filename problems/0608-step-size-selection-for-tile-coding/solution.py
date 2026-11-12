import numpy as np

def tile_coding_step(base_alpha: float, n_tilings: int, weights: list, active_tiles: list, target: float) -> dict:
    """
    Perform a single value function update using tile coding with adjusted step size.
    
    Args:
        base_alpha: The base learning rate before adjustment
        n_tilings: Number of tilings used in the tile coding scheme
        weights: List of current weight values for all tiles
        active_tiles: List of indices of tiles active for the current state
        target: The target value (e.g., a return or TD target)
    
    Returns:
        Dictionary with 'adjusted_alpha', 'prediction_before',
        'prediction_after', and 'updated_weights'
    """
    # 1. Compute the adjusted step size
    adjusted_alpha = base_alpha / n_tilings
    
    # 2. Compute the current value estimate before the update
    prediction_before = sum(weights[i] for i in active_tiles)
    
    # 3. Calculate the prediction error
    error = target - prediction_before
    
    # 4. Update weights for active tiles only
    new_weights = list(weights)
    for i in active_tiles:
        new_weights[i] += adjusted_alpha * error
        
    # 5. Compute the value estimate after the update
    prediction_after = sum(new_weights[i] for i in active_tiles)
    
    # Return dictionary with all numerical values rounded to 4 decimal places
    return {
        'adjusted_alpha': round(float(adjusted_alpha), 4),
        'prediction_before': round(float(prediction_before), 4),
        'prediction_after': round(float(prediction_after), 4),
        'updated_weights': [round(float(w), 4) for w in new_weights]
    }