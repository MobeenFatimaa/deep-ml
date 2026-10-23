import numpy as np
import math

def llama3_rope_rescale(inv_freq, original_context_length, low_freq_factor, high_freq_factor, scaling_factor):
    """
    Apply Llama 3.1 style piecewise rescaling to a set of RoPE inverse frequencies.

    Args:
        inv_freq: list or array of base inverse frequencies
        original_context_length: int, the original context length used during pretraining
        low_freq_factor: float, threshold factor defining the low-frequency cutoff
        high_freq_factor: float, threshold factor defining the high-frequency cutoff
        scaling_factor: float, the factor by which low-frequency components are divided

    Returns:
        list of floats: adjusted inverse frequencies, each rounded to 6 decimals
    """
    low_freq_wavelen = original_context_length / low_freq_factor
    high_freq_wavelen = original_context_length / high_freq_factor

    adjusted_freqs = []
    for freq in inv_freq:
        wavelen = 2 * math.pi / freq
        
        if wavelen > low_freq_wavelen:
            new_freq = freq / scaling_factor
        elif wavelen < high_freq_wavelen:
            new_freq = freq
        else:
            smooth = (original_context_length / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freq = (1 - smooth) * (freq / scaling_factor) + smooth * freq
            
        adjusted_freqs.append(round(float(new_freq), 6))

    return adjusted_freqs