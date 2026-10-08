import math


def capability_gate(
    n_params: float,
    n_tokens: float,
    log10_compute_thresholds: list[float],
    eval_scores: dict[str, float],
    eval_limits: dict[str, float],
) -> dict:
    """Account training FLOPs and return a compute / eval capability gate decision."""
    # 1. Compute total training FLOPs: C = 6 * n_params * n_tokens
    compute_flops = 6.0 * float(n_params) * float(n_tokens)

    # 2. Compute L = log10(C) in full precision
    unrounded_L = math.log10(compute_flops)
    log10_flops = round(unrounded_L, 4)

    # 3. Determine compute band (count thresholds where unrounded L >= threshold)
    compute_band = sum(1 for t in log10_compute_thresholds if unrounded_L >= t)

    # 4. Find flagged evaluations (common evals where score >= limit)
    flagged = [
        name
        for name in eval_scores
        if name in eval_limits and eval_scores[name] >= eval_limits[name]
    ]
    flagged_evals = sorted(flagged)

    # 5. Determine responsible-scaling gate decision
    if len(flagged_evals) > 0 or compute_band >= 2:
        decision = "pause"
    elif compute_band == 1 and len(flagged_evals) == 0:
        decision = "report"
    else:
        decision = "below"

    return {
        "log10_flops": log10_flops,
        "compute_band": compute_band,
        "flagged_evals": flagged_evals,
        "decision": decision,
    }