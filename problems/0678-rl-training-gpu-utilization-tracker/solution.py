def track_gpu_utilization(timestamps_ms: list, gpu_util_pct: list, phase_labels: list) -> dict:
    """
    Analyze GPU utilization during RL training and compute per-phase statistics.
    
    Args:
        timestamps_ms: List of N sorted timestamps (ms) marking interval boundaries.
        gpu_util_pct: List of N-1 GPU utilization percentages (0-100) per interval.
        phase_labels: List of N-1 phase labels per interval.
    
    Returns:
        dict with keys: total_time_ms, avg_gpu_util_pct, phase_stats,
                        bottleneck_phase, gpu_idle_fraction_pct
    """
    if not timestamps_ms or len(timestamps_ms) < 2:
        return {
            'total_time_ms': 0.0,
            'avg_gpu_util_pct': 0.0,
            'phase_stats': {},
            'bottleneck_phase': None,
            'gpu_idle_fraction_pct': 100.0
        }
        
    total_time = timestamps_ms[-1] - timestamps_ms[0]
    
    phase_data = {}
    total_weighted_util = 0.0
    
    for i in range(len(gpu_util_pct)):
        duration = timestamps_ms[i + 1] - timestamps_ms[i]
        util = gpu_util_pct[i]
        phase = phase_labels[i]
        
        if phase not in phase_data:
            phase_data[phase] = {
                'total_time': 0.0,
                'weighted_util': 0.0,
                'wasted_time': 0.0,
                'first_index': i
            }
            
        phase_data[phase]['total_time'] += duration
        phase_data[phase]['weighted_util'] += duration * util
        phase_data[phase]['wasted_time'] += duration * (100.0 - util)
        
        if i < phase_data[phase]['first_index']:
            phase_data[phase]['first_index'] = i
            
        total_weighted_util += duration * util
        
    avg_gpu_util_pct = round(total_weighted_util / total_time, 2) if total_time > 0 else 0.0
    gpu_idle_fraction_pct = round(100.0 - avg_gpu_util_pct, 2)
    
    sorted_phases = sorted(phase_data.keys())
    phase_stats = {}
    
    for phase in sorted_phases:
        p_info = phase_data[phase]
        p_time = p_info['total_time']
        p_avg_util = round(p_info['weighted_util'] / p_time, 2) if p_time > 0 else 0.0
        p_time_fraction = round((p_time / total_time) * 100, 2) if total_time > 0 else 0.0
        
        phase_stats[phase] = {
            'total_time_ms': round(p_time, 2),
            'avg_util_pct': p_avg_util,
            'time_fraction_pct': p_time_fraction
        }
        
    bottleneck_phase = None
    max_wasted = -1.0
    min_first_index = float('inf')
    
    for phase in sorted_phases:
        p_info = phase_data[phase]
        wasted = p_info['wasted_time']
        f_idx = p_info['first_index']
        
        if wasted > max_wasted:
            max_wasted = wasted
            min_first_index = f_idx
            bottleneck_phase = phase
        elif wasted == max_wasted:
            if f_idx < min_first_index:
                min_first_index = f_idx
                bottleneck_phase = phase
                
    return {
        'total_time_ms': round(total_time, 2),
        'avg_gpu_util_pct': avg_gpu_util_pct,
        'phase_stats': phase_stats,
        'bottleneck_phase': bottleneck_phase,
        'gpu_idle_fraction_pct': gpu_idle_fraction_pct
    }