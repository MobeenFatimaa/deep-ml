import numpy as np

def goal_conditioned_rollout(
    initial_state: np.ndarray,
    actions: np.ndarray,
    goal_state: np.ndarray,
    transition_matrix: np.ndarray,
    action_matrix: np.ndarray,
    goal_matrix: np.ndarray,
    anchor_strength: float = 0.5,
    anchor_power: float = 2.0
) -> dict:
    """
    Roll out a goal-conditioned world model with future anchoring.
    
    Anchor weight at step k (0-indexed) of T total steps:
        alpha_k = anchor_strength * ((k+1) / T) ^ anchor_power
    
    Args:
        initial_state: Shape (D,)
        actions: Shape (T, A) - sequence of T actions
        goal_state: Shape (D,) - desired goal state
        transition_matrix: Shape (D, D) - state transition weights
        action_matrix: Shape (D, A) - action influence weights
        goal_matrix: Shape (D, D) - goal conditioning weights
        anchor_strength: Max anchoring intensity in [0, 1]
        anchor_power: Controls anchoring ramp-up speed
    
    Returns:
        dict with keys:
        - 'trajectory': np.ndarray shape (T+1, D)
        - 'anchor_weights': np.ndarray shape (T,)
        - 'raw_predictions': np.ndarray shape (T, D)
    """
    T = actions.shape[0]
    D = initial_state.shape[0]
    
    trajectory = np.zeros((T + 1, D))
    trajectory[0] = initial_state
    
    anchor_weights = np.zeros(T)
    raw_predictions = np.zeros((T, D))
    
    for k in range(T):
        s_k = trajectory[k]
        a_k = actions[k]
        
        # Raw prediction: r_k = tanh(W_s * s_k + W_a * a_k + W_g * (g - s_k))
        pre_activation = transition_matrix @ s_k + action_matrix @ a_k + goal_matrix @ (goal_state - s_k)
        r_k = np.tanh(pre_activation)
        raw_predictions[k] = r_k
        
        # Anchor weight: alpha_k = anchor_strength * ((k+1) / T) ^ anchor_power
        alpha_k = anchor_strength * (((k + 1) / T) ** anchor_power)
        anchor_weights[k] = alpha_k
        
        # Next state: s_{k+1} = (1 - alpha_k) * r_k + alpha_k * g
        s_next = (1.0 - alpha_k) * r_k + alpha_k * goal_state
        trajectory[k + 1] = s_next
        
    return {
        'trajectory': trajectory,
        'anchor_weights': anchor_weights,
        'raw_predictions': raw_predictions
    }


def compute_future_anchor_loss(
    trajectory: np.ndarray,
    goal_state: np.ndarray,
    raw_predictions: np.ndarray,
    lambda_goal: float = 1.0,
    lambda_smooth: float = 0.1,
    lambda_consistency: float = 0.1
) -> dict:
    """
    Compute multi-component loss for the goal-conditioned world model.
    
    Args:
        trajectory: Shape (T+1, D)
        goal_state: Shape (D,)
        raw_predictions: Shape (T, D)
        lambda_goal: Weight for goal-reaching loss
        lambda_smooth: Weight for smoothness loss
        lambda_consistency: Weight for consistency loss
    
    Returns:
        dict with 'goal_loss', 'smoothness_loss', 'consistency_loss', 'total_loss'
    """
    T = raw_predictions.shape[0]
    
    # 1. Goal loss: ||s_T - g||^2
    s_T = trajectory[-1]
    goal_loss = float(np.sum((s_T - goal_state) ** 2))
    
    # 2. Smoothness loss: (1 / T) * sum_{k=1}^T ||s_k - s_{k-1}||^2
    diffs = trajectory[1:] - trajectory[:-1]
    smoothness_loss = float(np.sum(diffs ** 2) / T)
    
    # 3. Consistency loss: (1 / T) * sum_{k=0}^{T-1} ||r_k - s_{k+1}||^2
    next_states = trajectory[1:]
    consistency_diffs = raw_predictions - next_states
    consistency_loss = float(np.sum(consistency_diffs ** 2) / T)
    
    # 4. Total loss
    total_loss = (
        lambda_goal * goal_loss +
        lambda_smooth * smoothness_loss +
        lambda_consistency * consistency_loss
    )
    
    return {
        'goal_loss': goal_loss,
        'smoothness_loss': smoothness_loss,
        'consistency_loss': consistency_loss,
        'total_loss': total_loss
    }