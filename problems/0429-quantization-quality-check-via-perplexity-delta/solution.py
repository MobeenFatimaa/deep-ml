import numpy as np

def quantization_quality_check(log_probs_original: list, log_probs_quantized: list, thresholds: dict = None) -> dict:
    """
    Evaluate quantization quality by comparing perplexity before and after quantization.
    
    Args:
        log_probs_original: Token-level log-probabilities from the original (full precision) model
        log_probs_quantized: Token-level log-probabilities from the quantized model
        thresholds: Optional dict with 'excellent' and 'acceptable' percentage thresholds
    
    Returns:
        dict with perplexity metrics and quality classification
    """
    if thresholds is None:
        thresholds = {'excellent': 1.0, 'acceptable': 5.0}
    else:
        thresholds = {'excellent': 1.0, 'acceptable': 5.0, **thresholds}

    log_probs_orig_arr = np.asarray(log_probs_original, dtype=np.float64)
    log_probs_quant_arr = np.asarray(log_probs_quantized, dtype=np.float64)

    # Perplexity is exp of the mean negative log-likelihood
    perp_orig = float(np.exp(-np.mean(log_probs_orig_arr)))
    perp_quant = float(np.exp(-np.mean(log_probs_quant_arr)))

    perp_delta = float(perp_quant - perp_orig)
    relative_delta_percent = float((perp_delta / perp_orig) * 100.0)

    if relative_delta_percent <= thresholds['excellent']:
        quality = 'excellent'
    elif relative_delta_percent <= thresholds['acceptable']:
        quality = 'acceptable'
    else:
        quality = 'poor'

    return {
        'perplexity_original': round(perp_orig, 4),
        'perplexity_quantized': round(perp_quant, 4),
        'perplexity_delta': round(perp_delta, 4),
        'relative_delta_percent': round(relative_delta_percent, 4),
        'quality': quality
    }