def analyze_canary_deployment(
    canary_results: list,
    baseline_results: list,
    accuracy_tolerance: float = 0.05,
    latency_tolerance: float = 0.10,
) -> dict:
    """Analyze canary deployment health metrics for model rollout decision."""
    # Return empty dict if either list is empty
    if not canary_results or not baseline_results:
        return {}

    # Calculate Accuracy for Canary and Baseline
    canary_correct = sum(
        1 for r in canary_results if r["prediction"] == r["ground_truth"]
    )
    baseline_correct = sum(
        1 for r in baseline_results if r["prediction"] == r["ground_truth"]
    )

    canary_acc = canary_correct / len(canary_results)
    baseline_acc = baseline_correct / len(baseline_results)

    # Relative accuracy change percentage
    if baseline_acc > 0:
        accuracy_change_pct = (
            (canary_acc - baseline_acc) / baseline_acc
        ) * 100.0
    else:
        accuracy_change_pct = 0.0

    # Calculate Average Latency for Canary and Baseline
    canary_avg_lat = sum(r["latency_ms"] for r in canary_results) / len(
        canary_results
    )
    baseline_avg_lat = sum(r["latency_ms"] for r in baseline_results) / len(
        baseline_results
    )

    # Relative latency change percentage
    if baseline_avg_lat > 0:
        latency_change_pct = (
            (canary_avg_lat - baseline_avg_lat) / baseline_avg_lat
        ) * 100.0
    else:
        latency_change_pct = 0.0

    # Check promotion tolerance conditions:
    # 1. Accuracy degradation does not exceed accuracy_tolerance (e.g. -accuracy_change_pct <= accuracy_tolerance * 100)
    # 2. Latency increase does not exceed latency_tolerance (e.g. latency_change_pct <= latency_tolerance * 100)
    accuracy_degradation_pct = (
        -accuracy_change_pct
    )  # positive value indicates a drop
    latency_increase_pct = (
        latency_change_pct  # positive value indicates a rise
    )

    promote_recommended = (
        accuracy_degradation_pct <= (accuracy_tolerance * 100.0)
    ) and (latency_increase_pct <= (latency_tolerance * 100.0))

    return {
        "canary_accuracy": round(canary_acc, 4),
        "baseline_accuracy": round(baseline_acc, 4),
        "accuracy_change_pct": round(accuracy_change_pct, 2),
        "canary_avg_latency": round(canary_avg_lat, 2),
        "baseline_avg_latency": round(baseline_avg_lat, 2),
        "latency_change_pct": round(latency_change_pct, 2),
        "promote_recommended": promote_recommended,
    }