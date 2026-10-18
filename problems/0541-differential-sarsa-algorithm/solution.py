def differential_sarsa(
    transitions: dict,
    initial_state: str,
    alpha: float,
    beta: float,
    num_steps: int,
) -> tuple:
    """Differential Sarsa for the average-reward continuing setting.

    Args:
        transitions: dict mapping (state, action) -> (reward, next_state)
        initial_state: starting state
        alpha: step size for Q-value updates
        beta: step size for average reward estimate
        num_steps: number of steps to simulate

    Returns:
        Tuple of (Q, R_bar) where Q is a dict {(state, action): float}
        and R_bar is a float.
    """
    # Initialize Q-values to zero for all state-action pairs in transitions
    Q = {sa: 0.0 for sa in transitions.keys()}
    R_bar = 0.0

    def get_best_action(state):
        available_actions = [a for (s, a) in transitions.keys() if s == state]
        best_action = None
        max_q = float("-inf")
        for a in sorted(available_actions):
            q_val = Q.get((state, a), 0.0)
            if q_val > max_q:
                max_q = q_val
                best_action = a
        return best_action

    current_state = initial_state
    current_action = get_best_action(current_state)

    for _ in range(num_steps):
        reward, next_state = transitions[(current_state, current_action)]
        next_action = get_best_action(next_state)

        # Differential TD error
        delta = (
            reward
            - R_bar
            + Q.get((next_state, next_action), 0.0)
            - Q.get((current_state, current_action), 0.0)
        )

        # Update average reward estimate
        R_bar += beta * delta

        # Update Q-value
        Q[(current_state, current_action)] += alpha * delta

        # Transition to next state and action
        current_state = next_state
        current_action = next_action

    return Q, R_bar