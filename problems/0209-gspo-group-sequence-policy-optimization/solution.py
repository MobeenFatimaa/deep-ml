import numpy as np


def gspo_objective(
    log_probs_new: list[list[float]],
    log_probs_old: list[list[float]],
    rewards: list[float],
    epsilon: float = 0.2,
) -> float:
    """Compute GSPO (Group Sequence Policy Optimization) clipped objective."""
    log_probs_new = [np.array(seq) for seq in log_probs_new]
    log_probs_old = [np.array(seq) for seq in log_probs_old]
    rewards = np.array(rewards)

    mean_r = np.mean(rewards)
    std_r = np.std(rewards)
    advantages = (rewards - mean_r) / (std_r if std_r > 0 else 1.0)

    objectives = []
    for i in range(len(rewards)):
        diff = log_probs_new[i] - log_probs_old[i]
        length = len(diff)
        ratio = np.exp(np.sum(diff) / length)

        clipped_ratio = np.clip(ratio, 1.0 - epsilon, 1.0 + epsilon)
        obj = min(ratio * advantages[i], clipped_ratio * advantages[i])
        objectives.append(obj)

    return float(np.mean(objectives))