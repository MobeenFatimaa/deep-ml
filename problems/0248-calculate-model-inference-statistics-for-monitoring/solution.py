import numpy as np

def calculate_inference_stats(latencies_ms: list[float]) -> dict[str, float]:
    """Calculate key inference performance statistics for monitoring.

    Args:
        latencies_ms: List of inference latency measurements in milliseconds.

    Returns:
        Dictionary containing throughput, avg latency, p50, p95, and p99.
    """
    if not latencies_ms:
        return {}

    total_requests = len(latencies_ms)
    avg_latency = float(np.mean(latencies_ms))
    
    # Throughput = requests / total_time_in_seconds
    # total_time_in_seconds = sum(latencies_ms) / 1000
    total_time_sec = sum(latencies_ms) / 1000.0
    
    # Avoid division by zero if all latencies are 0.0
    throughput = (total_requests / total_time_sec) if total_time_sec > 0 else float('inf')

    # Compute percentiles using linear interpolation (method='linear' is default in np.percentile)
    p50 = float(np.percentile(latencies_ms, 50, method='linear'))
    p95 = float(np.percentile(latencies_ms, 95, method='linear'))
    p99 = float(np.percentile(latencies_ms, 99, method='linear'))

    return {
        'throughput_per_sec': round(throughput, 2),
        'avg_latency_ms': round(avg_latency, 2),
        'p50_ms': round(p50, 2),
        'p95_ms': round(p95, 2),
        'p99_ms': round(p99, 2)
    }