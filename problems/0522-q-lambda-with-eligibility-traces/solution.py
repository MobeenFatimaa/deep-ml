
import numpy as np

def q_lambda_watkins(episodes: list, n_states: int, n_actions: int, gamma: float, alpha: float, lam: float) -> list:
    """
    Implement Watkins's Q(lambda) with accumulating eligibility traces.
    """
    Q = np.zeros((n_states, n_actions), dtype=float)

    for episode in episodes:
        eligibility = np.zeros((n_states, n_actions))

        for state, action, reward, next_state, done in episode:

            # Compute the TD error
            if done:
                delta = reward - Q[state, action]
            else:
                # Greedy action selection with lowest-index tie-breaking
                greedy_next_action = int(np.argmax(Q[next_state]))
                delta = (
                    reward
                    + gamma * Q[next_state, greedy_next_action]
                    - Q[state, action]
                )

            # Accumulating eligibility trace
            eligibility[state, action] += 1.0

            # Update all state-action pairs
            Q += alpha * delta * eligibility

            # Terminal transitions end the episode
            if done:
                break

            # Check whether the behavior action at the next state
            # is greedy under the updated Q-table.
            next_transition_index = episode.index(
                (state, action, reward, next_state, done)
            ) + 1

            if next_transition_index < len(episode):
                next_action = episode[next_transition_index][1]
                greedy_action = int(np.argmax(Q[next_state]))

                if next_action != greedy_action:
                    # Watkins's trace cutting
                    eligibility.fill(0.0)
                else:
                    # Decay traces when the next action is greedy
                    eligibility *= gamma * lam
            else:
                eligibility *= gamma * lam

    return np.round(Q, 4).tolist()
