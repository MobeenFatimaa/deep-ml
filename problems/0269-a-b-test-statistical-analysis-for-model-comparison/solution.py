import math
from scipy import stats


def analyze_ab_test(
    control_outcomes: list,
    treatment_outcomes: list,
    confidence_level: float = 0.95,
    min_detectable_effect: float = 0.02,
) -> dict:
    """Analyze A/B test results for model comparison with statistical rigor."""
    # Empty input handling
    if not control_outcomes or not treatment_outcomes:
        return {}

    n_control = len(control_outcomes)
    n_treatment = len(treatment_outcomes)

    if n_control == 0 or n_treatment == 0:
        return {}

    # 1. Success Rates
    p_control = sum(control_outcomes) / n_control
    p_treatment = sum(treatment_outcomes) / n_treatment

    # 2. Lift Metrics
    absolute_lift = p_treatment - p_control
    relative_lift_pct = (
        (absolute_lift / p_control) * 100.0 if p_control > 0 else 0.0
    )

    # 3. Two-Proportion Z-Test (Pooled Variance)
    total_successes = sum(control_outcomes) + sum(treatment_outcomes)
    total_n = n_control + n_treatment
    p_pooled = total_successes / total_n

    se_pooled = math.sqrt(
        p_pooled * (1.0 - p_pooled) * (1.0 / n_control + 1.0 / n_treatment)
    )

    if se_pooled > 0:
        z_stat = absolute_lift / se_pooled
    else:
        z_stat = 0.0

    # Two-tailed p-value
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))

    # 4. Confidence Interval (Unpooled Variance)
    alpha = 1.0 - confidence_level
    z_critical = stats.norm.ppf(1.0 - alpha / 2.0)

    se_unpooled = math.sqrt(
        (p_control * (1.0 - p_control) / n_control)
        + (p_treatment * (1.0 - p_treatment) / n_treatment)
    )

    ci_lower = absolute_lift - z_critical * se_unpooled
    ci_upper = absolute_lift + z_critical * se_unpooled

    # 5. Significance Checks
    statistically_significant = p_value < alpha
    practically_significant = abs(absolute_lift) >= min_detectable_effect

    # 6. Required Sample Size per Group for 80% Power (Power = 0.80 -> Z_beta = 0.8416)
    z_beta = stats.norm.ppf(0.80)
    mde = min_detectable_effect

    if mde > 0 and 0 < p_pooled < 1.0:
        req_sample_size = math.ceil(
            (
                2.0
                * ((z_critical + z_beta) ** 2)
                * p_pooled
                * (1.0 - p_pooled)
            )
            / (mde**2)
        )
    else:
        req_sample_size = 0

    # 7. Recommendation Logic
    if statistically_significant and practically_significant and (p_treatment > p_control):
        recommendation = "launch_treatment"
    elif statistically_significant and (
        (p_treatment <= p_control) or (not practically_significant)
    ):
        recommendation = "keep_control"
    else:
        recommendation = "continue_testing"

    return {
        "control_rate": round(p_control, 4),
        "treatment_rate": round(p_treatment, 4),
        "absolute_lift": round(absolute_lift, 4),
        "relative_lift_pct": round(relative_lift_pct, 2),
        "z_statistic": round(z_stat, 4),
        "p_value": round(p_value, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "statistically_significant": statistically_significant,
        "practically_significant": practically_significant,
        "required_sample_size": req_sample_size,
        "recommendation": recommendation,
    }
    