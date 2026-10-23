import numpy as np

def rollout_video_dynamics(z0: np.ndarray, actions: np.ndarray,
                           W_scale: np.ndarray, b_scale: np.ndarray,
                           W_shift: np.ndarray, b_shift: np.ndarray,
                           W_hidden: np.ndarray, b_hidden: np.ndarray,
                           W_out: np.ndarray, b_out: np.ndarray) -> np.ndarray:
    """
    Autoregressively roll out a video dynamics model conditioned on actions.

    Args:
        z0: Initial latent, shape (D,)
        actions: Action sequence, shape (T, A)
        W_scale, b_scale: Linear projection to per-feature scale (shape (A, D), (D,))
        W_shift, b_shift: Linear projection to per-feature shift (shape (A, D), (D,))
        W_hidden, b_hidden: Hidden transform weights (shape (D, D), (D,))
        W_out, b_out: Output transform weights (shape (D, D), (D,))

    Returns:
        Predicted latent sequence of shape (T, D)
    """
    z0 = np.array(z0, dtype=np.float64)
    actions = np.array(actions, dtype=np.float64)
    
    T = actions.shape[0]
    z_curr = z0.copy()
    trajectory = []

    for t in range(T):
        act = actions[t]
        scale = act @ W_scale + b_scale
        shift = act @ W_shift + b_shift
        
        z_mod = scale * z_curr + shift
        hidden = np.tanh(z_mod @ W_hidden + b_hidden)
        update = hidden @ W_out + b_out
        
        z_curr = z_curr + update
        trajectory.append(z_curr.copy())

    return np.array(trajectory, dtype=np.float64)