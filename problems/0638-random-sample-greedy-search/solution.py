import numpy as np


def random_sample_greedy_search(
    q_table: np.ndarray,
    states: list,
    sample_size: int,
    epsilon: float,
    seed: int,
) -> dict:
    """
    Perform Random Sample Greedy action selection for a batch of states.

    Args:
        q_table: np.ndarray of shape (num_states, num_actions), Q-value estimates
        states: list of int, state indices to select actions for
        sample_size: int, number of actions to sample for greedy selection
        epsilon: float, probability of selecting a uniformly random action
        seed: int, random seed for reproducibility

    Returns:
        dict with keys:
            'actions': list of int, selected action indices
            'q_values': list of float, Q-values for selected actions (rounded to 4 dp)
            'greedy_match_rate': float, fraction matching true greedy (rounded to 4 dp)
    """
    q_table = np.asarray(q_table, dtype=float)
    num_actions = q_table.shape[1]

    if len(states) == 0:
        return {"actions": [], "q_values": [], "greedy_match_rate": 0.0}

    k = max(1, min(sample_size, num_actions))
    rng = np.random.RandomState(seed)

    actions = []
    q_values = []
    greedy_matches = 0

    for state in states:
        row = q_table[state]
        # np.argmax returns the first (smallest) index on ties
        true_greedy = int(np.argmax(row))

        if rng.random() < epsilon:
            action = int(rng.randint(0, num_actions))
        else:
            sampled = np.sort(rng.choice(num_actions, size=k, replace=False))
            best_pos = int(np.argmax(row[sampled]))
            action = int(sampled[best_pos])

        if action == true_greedy:
            greedy_matches += 1

        actions.append(action)
        q_values.append(round(float(row[action]), 4))

    greedy_match_rate = round(float(greedy_matches / len(states)), 4)

    return {
        "actions": actions,
        "q_values": q_values,
        "greedy_match_rate": greedy_match_rate,
    }