import numpy as np

def build_alphazero_targets(trajectories):
    """
    Convert self-play trajectories into AlphaZero training tuples.

    Args:
        trajectories: list of dicts with keys 'states', 'visits', 'winner'.

    Returns:
        List of [state, pi_target, z_target] training tuples.
    """
    dataset = []

    for traj in trajectories:
        states = traj['states']
        visits = traj['visits']
        winner = float(traj['winner'])

        for step_idx, (state, visit_counts) in enumerate(zip(states, visits)):
            # 1. Normalize MCTS visit counts to form pi_target
            total_visits = sum(visit_counts)
            if total_visits > 0:
                pi_target = [v / total_visits for v in visit_counts]
            else:
                pi_target = [0.0] * len(visit_counts)

            # 2. Adjust z_target for alternating player perspectives
            # Even index: Player 1 moves -> z_target = winner
            # Odd index: Player 2 moves  -> z_target = -winner
            z_target = winner if step_idx % 2 == 0 else -winner

            dataset.append([state, pi_target, z_target])

    return dataset