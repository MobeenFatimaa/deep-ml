import numpy as np


def rl_budget_loss(
    rewards: np.ndarray,
    log_probs: np.ndarray,
    old_log_probs: np.ndarray,
    response_lengths: np.ndarray,
    token_budget: int,
    kl_coef: float,
    budget_penalty_coef: float,
) -> float:
    """
    Compute the budget-constrained RL loss.

    The loss combines:
    1. Budget penalty for responses exceeding token_budget
    2. Advantage estimation (adjusted reward - baseline)
    3. KL regularization between current and old policy

    Loss formula: E[(advantage - kl_term)^2]

    Args:
        rewards: Shape (batch_size, K) - rewards for K samples per prompt
        log_probs: Shape (batch_size, K) - log π_θ(y|x) current policy
        old_log_probs: Shape (batch_size, K) - log π_old(y|x) old policy
        response_lengths: Shape (batch_size, K) - token lengths of responses
        token_budget: Maximum allowed tokens before penalty
        kl_coef: τ coefficient for KL regularization
        budget_penalty_coef: λ coefficient for budget penalty

    Returns:
        Scalar loss value (float)
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    log_probs = np.asarray(log_probs, dtype=np.float64)
    old_log_probs = np.asarray(old_log_probs, dtype=np.float64)
    response_lengths = np.asarray(response_lengths, dtype=np.float64)

    excess_tokens = np.maximum(0.0, response_lengths - float(token_budget))
    penalties = -float(budget_penalty_coef) * excess_tokens
    adjusted_rewards = rewards + penalties

    baselines = np.mean(adjusted_rewards, axis=1, keepdims=True)
    advantages = adjusted_rewards - baselines

    kl_term = float(kl_coef) * (log_probs - old_log_probs)
    loss = np.mean((advantages - kl_term) ** 2)

    return float(loss)