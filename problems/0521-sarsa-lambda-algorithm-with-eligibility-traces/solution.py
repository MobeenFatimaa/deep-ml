
import numpy as np

def sarsa_lambda(
    n_states: int,
    n_actions: int,
    transitions: dict,
    terminal_states: list,
    gamma: float,
    alpha: float,
    lam: float,
    epsilon: float,
    num_episodes: int,
    seed: int = 42
) -> list:
    """
    Sarsa(lambda) with accumulating eligibility traces.
    """
    rng = np.random.RandomState(seed)

    Q = np.zeros((n_states, n_actions), dtype=float)
    terminal_set = set(terminal_states)
    non_terminal_states = [
        s for s in range(n_states) if s not in terminal_set
    ]

    if not non_terminal_states or num_episodes <= 0:
        return Q.round(4).tolist()

    def select_action(state):
        if rng.random() < epsilon:
            return int(rng.randint(n_actions))

        # Break ties by selecting the smallest action index.
        return int(np.argmax(Q[state]))

    for _ in range(num_episodes):
        eligibility = np.zeros((n_states, n_actions))

        state = int(rng.choice(non_terminal_states))
        action = select_action(state)

        while state not in terminal_set:
            if (state, action) not in transitions:
                break

            next_state, reward = transitions[(state, action)]

            if next_state in terminal_set:
                delta = reward - Q[state, action]
            else:
                next_action = select_action(next_state)
                delta = (
                    reward
                    + gamma * Q[next_state, next_action]
                    - Q[state, action]
                )

            # Accumulating eligibility trace
            eligibility[state, action] += 1.0

            # Update Q-values using current eligibility traces
            Q += alpha * delta * eligibility

            # Decay traces after the update
            eligibility *= gamma * lam

            if next_state in terminal_set:
                break

            state = next_state
            action = next_action

        # Terminal-state Q-values must remain zero
        for terminal in terminal_set:
            Q[terminal, :] = 0.0

    return np.round(Q, 4).tolist()