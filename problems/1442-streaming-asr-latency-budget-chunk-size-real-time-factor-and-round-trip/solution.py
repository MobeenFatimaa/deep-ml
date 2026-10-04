import math


def chunk_latency(chunk_s, rtf, rtt_s):
    """
    Calculate total latency to process a single chunk of audio.
    Latency = Wait time to record chunk + Model inference time + Network Round-Trip Time
    """
    return chunk_s + rtf * chunk_s + rtt_s


def largest_chunk_under(budget_s, rtf, rtt_s, step=0.05):
    """
    Find the largest multiple of step (>= 1 * step) such that chunk_latency <= budget_s.
    Returns None if even a single step exceeds the budget.
    """
    best_chunk = None
    k = 1
    while True:
        candidate_chunk = k * step
        if chunk_latency(candidate_chunk, rtf, rtt_s) <= budget_s + 1e-12:
            best_chunk = candidate_chunk
            k += 1
        else:
            break

    return round(best_chunk, 2) if best_chunk is not None else None


def concurrent_streams(rtf, gpu_share=1.0):
    """
    Calculate the max concurrent streams a GPU can serve: floor(gpu_share / rtf).
    """
    if rtf <= 0:
        return 0
    return math.floor(gpu_share / rtf)


def end_of_speech_latency(hangover_s, chunk_s, rtf, rtt_s):
    """
    Calculate latency from user's last spoken word to final text:
    hangover_s + chunk_latency(chunk_s, rtf, rtt_s)
    """
    return hangover_s + chunk_latency(chunk_s, rtf, rtt_s)


def budget_breakdown(chunk_s, rtf, rtt_s):
    """
    Return the fractional breakdown of chunk_latency across:
    - wait: chunk_s / total
    - model: (rtf * chunk_s) / total
    - network: rtt_s / total
    """
    total = chunk_latency(chunk_s, rtf, rtt_s)
    if total <= 0:
        return {"wait": 0.0, "model": 0.0, "network": 0.0}

    return {
        "wait": round(chunk_s / total, 4),
        "model": round((rtf * chunk_s) / total, 4),
        "network": round(rtt_s / total, 4),
    }