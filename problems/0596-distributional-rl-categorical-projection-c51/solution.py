import numpy as np

def categorical_projection(current_probs, next_probs, reward, gamma, done, v_min, v_max, n_atoms):
    """
    Perform categorical distributional Bellman projection and compute cross-entropy loss.
    
    Args:
        current_probs: np.ndarray of shape (n_atoms,), current predicted distribution
        next_probs: np.ndarray of shape (n_atoms,), next-state return distribution
        reward: float, observed reward
        gamma: float, discount factor
        done: int, 1 if terminal else 0
        v_min: float, minimum support value
        v_max: float, maximum support value
        n_atoms: int, number of support atoms
    
    Returns:
        tuple: (projected, loss) - projected target distribution and cross-entropy loss
    """
    current_probs = np.asarray(current_probs, dtype=np.float64)
    next_probs = np.asarray(next_probs, dtype=np.float64)
    
    support = np.linspace(v_min, v_max, n_atoms)
    delta_z = (v_max - v_min) / (n_atoms - 1) if n_atoms > 1 else 1.0
    
    projected = np.zeros(n_atoms, dtype=np.float64)
    
    # Compute Bellman update for each support atom
    Tz = reward + (1 - done) * gamma * support
    Tz = np.clip(Tz, v_min, v_max)
    
    # Compute projection positions
    b = (Tz - v_min) / delta_z
    l = np.floor(b).astype(np.int64)
    u = np.ceil(b).astype(np.int64)
    
    for j in range(n_atoms):
        # Handle case where l == u (e.g. exactly at the boundary or integer index)
        if l[j] == u[j]:
            projected[l[j]] += next_probs[j]
        else:
            projected[l[j]] += next_probs[j] * (u[j] - b[j])
            projected[u[j]] += next_probs[j] * (b[j] - l[j])
            
    # Cross-entropy loss with clipping on predicted probabilities to avoid log(0)
    clipped_current = np.clip(current_probs, 1e-8, 1.0)
    loss = -np.sum(projected * np.log(clipped_current))
    
    return projected, float(loss)