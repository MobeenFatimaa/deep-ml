import numpy as np

def disaggregated_serving_sim(requests, num_prefill, num_decode, prefill_rate, decode_rate, kv_transfer_rate):
    """
    Simulate disaggregated prefill-decode LLM serving.
    
    Args:
        requests: List of dicts with 'arrival_time', 'prompt_tokens', 'output_tokens'
        num_prefill: Number of prefill GPU instances
        num_decode: Number of decode GPU instances
        prefill_rate: Tokens/second per prefill instance
        decode_rate: Tokens/second per decode instance
        kv_transfer_rate: Seconds per prompt token for KV cache transfer
    
    Returns:
        Dict with 'avg_ttft', 'avg_total_latency', 'throughput',
             'prefill_utilization', 'decode_utilization'
    """
    if not requests:
        return {
            'avg_ttft': 0.0,
            'avg_total_latency': 0.0,
            'throughput': 0.0,
            'prefill_utilization': 0.0,
            'decode_utilization': 0.0
        }

    # Sort requests by arrival time to process in FCFS order
    sorted_reqs = sorted(requests, key=lambda x: x['arrival_time'])

    prefill_avail = [0.0] * num_prefill
    decode_avail = [0.0] * num_decode

    ttfts = []
    latencies = []
    total_output_tokens = 0
    total_prefill_busy = 0.0
    total_decode_busy = 0.0
    
    min_arrival = sorted_reqs[0]['arrival_time']
    max_completion = 0.0

    for req in sorted_reqs:
        arr = req['arrival_time']
        prompt_toks = req['prompt_tokens']
        output_toks = req['output_tokens']
        total_output_tokens += output_toks

        # 1. Prefill stage assignment: pick prefill instance with earliest availability
        p_idx = 0
        min_p_avail = prefill_avail[0]
        for i in range(1, num_prefill):
            if prefill_avail[i] < min_p_avail:
                min_p_avail = prefill_avail[i]
                p_idx = i

        prefill_start = max(arr, prefill_avail[p_idx])
        prefill_duration = prompt_toks / prefill_rate
        prefill_end = prefill_start + prefill_duration
        prefill_avail[p_idx] = prefill_end
        total_prefill_busy += prefill_duration

        # 2. KV Transfer stage
        kv_transfer_time = prompt_toks * kv_transfer_rate
        transfer_end = prefill_end + kv_transfer_time

        # 3. Decode stage assignment: pick decode instance with earliest availability
        d_idx = 0
        min_d_avail = decode_avail[0]
        for i in range(1, num_decode):
            if decode_avail[i] < min_d_avail:
                min_d_avail = decode_avail[i]
                d_idx = i

        decode_start = max(transfer_end, decode_avail[d_idx])
        decode_duration = output_toks / decode_rate
        decode_end = decode_start + decode_duration
        decode_avail[d_idx] = decode_end
        total_decode_busy += decode_duration

        if decode_end > max_completion:
            max_completion = decode_end

        # Metrics per request
        ttft = decode_start - arr
        total_latency = decode_end - arr
        ttfts.append(ttft)
        latencies.append(total_latency)

    makespan = max_completion - min_arrival
    if makespan == 0:
        makespan = 1e-9 # Prevent division by zero

    avg_ttft = float(np.mean(ttfts))
    avg_total_latency = float(np.mean(latencies))
    throughput = float(total_output_tokens / makespan)
    prefill_utilization = float(total_prefill_busy / (num_prefill * makespan))
    decode_utilization = float(total_decode_busy / (num_decode * makespan))

    return {
        'avg_ttft': round(avg_ttft, 4),
        'avg_total_latency': round(avg_total_latency, 4),
        'throughput': round(throughput, 4),
        'prefill_utilization': round(prefill_utilization, 4),
        'decode_utilization': round(decode_utilization, 4)
    }