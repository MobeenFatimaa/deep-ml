import numpy as np


def gambler_value_iteration(ph, theta=1e-9):
    """Computes the optimal value function and policy for the Gambler's Problem.

    Args:
        ph: probability of heads
        theta: convergence threshold

    Returns:
        V: list of values for all states 0..100
        policy: list of optimal stakes for all states 0..100
    """
    goal = 100
    V = np.zeros(goal + 1)

    while True:
        delta = 0
        for s in range(1, goal):
            old_v = V[s]
            actions = range(1, min(s, goal - s) + 1)
            best_v = -float("inf")
            for a in actions:
                win_val = 1.0 if (s + a == goal) else V[s + a]
                lose_val = V[s - a]
                v = ph * win_val + (1 - ph) * lose_val
                if v > best_v:
                    best_v = v
            V[s] = best_v
            delta = max(delta, abs(old_v - V[s]))

        if delta < theta:
            break

    policy = np.zeros(goal + 1, dtype=int)
    for s in range(1, goal):
        actions = range(1, min(s, goal - s) + 1)
        best_v = -float("inf")
        best_a = 1
        for a in actions:
            win_val = 1.0 if (s + a == goal) else V[s + a]
            lose_val = V[s - a]
            v = ph * win_val + (1 - ph) * lose_val
            if v > best_v + 1e-11:
                best_v = v
                best_a = a
        policy[s] = best_a

    return V.tolist(), policy.tolist()