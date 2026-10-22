import numpy as np

def option_value_iteration(R: np.ndarray, P: np.ndarray, options: list, gamma: float, num_iterations: int) -> dict:
    """
    Compute value functions for options using iterative Bellman updates.
    
    Args:
        R: Reward matrix of shape (S, A)
        P: Transition probability tensor of shape (S, A, S)
        options: List of option dicts with 'policy' (S, A) and 'termination' (S,)
        gamma: Discount factor
        num_iterations: Number of iteration sweeps
    
    Returns:
        Dictionary with 'Q_options', 'V', and 'U'
    """
    S, A = R.shape
    num_options = len(options)
    
    # Initialize Q_options(s, o) = 0
    Q_options = np.zeros((S, num_options))
    
    for _ in range(num_iterations):
        # 1. Compute V_O(s) = max_o Q_O(s, o)
        V = np.max(Q_options, axis=1) # shape (S,)
        
        # 2. Compute Q_options using synchronous update
        new_Q_options = np.zeros((S, num_options))
        for o_idx, option in enumerate(options):
            pi_o = np.array(option['policy']) # shape (S, A)
            beta_o = np.array(option['termination']) # shape (S,)
            
            # U(o, s') = (1 - beta_o(s')) * Q_O(s', o) + beta_o(s') * V(s')
            # Q_O shape is (S, num_options), so Q_options[:, o_idx] is Q_O(s', o) for all s'
            U_o = (1.0 - beta_o) * Q_options[:, o_idx] + beta_o * V # shape (S,)
            
            # Q_O(s, o) = sum_a pi_o(a | s) [ R(s, a) + gamma * sum_{s'} P(s, a, s') U(o, s') ]
            # P is (S, A, S), U_o is (S,) -> sum_{s'} P(s, a, s') U_o(s') has shape (S, A)
            expected_future = np.sum(P * U_o, axis=2) # shape (S, A)
            # R + gamma * expected_future has shape (S, A)
            q_sum_a = pi_o * (R + gamma * expected_future) # shape (S, A)
            new_Q_options[:, o_idx] = np.sum(q_sum_a, axis=1) # shape (S,)
            
        Q_options = new_Q_options
        
    # After the last sweep, recompute V and U from final Q_options
    V = np.max(Q_options, axis=1)
    U = np.zeros((num_options, S))
    for o_idx, option in enumerate(options):
        beta_o = np.array(option['termination'])
        U[o_idx, :] = (1.0 - beta_o) * Q_options[:, o_idx] + beta_o * V
        
    return {
        'Q_options': np.round(Q_options, 4).tolist(),
        'V': np.round(V, 4).tolist(),
        'U': np.round(U, 4).tolist()
    }