import math

def analyze_array_performance(n: int, operations: list) -> dict:
    """
    Analyze and compare array operation performance across Python, Cython, and NumPy.
    
    Args:
        n: Array size (number of elements)
        operations: List of dicts with performance parameters for each operation
    
    Returns:
        dict with per-operation metrics, pipeline totals, recommendation,
        and crossover sizes.
    """
    ops_result = []
    crossover_sizes = {}
    
    total_py = 0.0
    total_cy = 0.0
    total_np = 0.0
    
    priority = {'cython': 0, 'numpy': 1, 'python': 2}
    
    for op in operations:
        name = op['name']
        py_pe = op['python_per_elem_ns']
        cy_pe = op['cython_per_elem_ns']
        np_pe = op['numpy_per_elem_ns']
        
        py_oh = op['python_overhead_ns']
        cy_oh = op['cython_overhead_ns']
        np_oh = op['numpy_overhead_ns']
        
        py_time = (py_oh + n * py_pe) / 1000.0
        cy_time = (cy_oh + n * cy_pe) / 1000.0
        np_time = (np_oh + n * np_pe) / 1000.0
        
        py_us = round(py_time, 2)
        cy_us = round(cy_time, 2)
        np_us = round(np_time, 2)
        
        total_py += py_time
        total_cy += cy_time
        total_np += np_time
        
        cython_speedup = round(py_time / cy_time, 2) if cy_time > 0 else 0.0
        numpy_speedup = round(py_time / np_time, 2) if np_time > 0 else 0.0
        
        # Determine best method with tie-breaking: cython < numpy < python
        times = {'cython': cy_time, 'numpy': np_time, 'python': py_time}
        best_method = min(times.keys(), key=lambda k: (times[k], priority[k]))
        
        ops_result.append({
            'name': name,
            'python_us': py_us,
            'cython_us': cy_us,
            'numpy_us': np_us,
            'cython_speedup': cython_speedup,
            'numpy_speedup': numpy_speedup,
            'best': best_method
        })
        
        # Crossover size calculation for NumPy vs Cython
        if cy_pe <= np_pe:
            crossover = -1
        else:
            ratio = (np_oh - cy_oh) / (cy_pe - np_pe)
            if ratio <= 0:
                crossover = 0
            else:
                crossover = math.ceil(ratio)
        
        crossover_sizes[name] = crossover

    pipeline_python_us = round(total_py, 2)
    pipeline_cython_us = round(total_cy, 2)
    pipeline_numpy_us = round(total_np, 2)
    
    totals = {'cython': total_cy, 'numpy': total_np, 'python': total_py}
    recommended = min(totals.keys(), key=lambda k: (totals[k], priority[k]))
    
    return {
        'operations': ops_result,
        'pipeline_python_us': pipeline_python_us,
        'pipeline_cython_us': pipeline_cython_us,
        'pipeline_numpy_us': pipeline_numpy_us,
        'recommended': recommended,
        'crossover_sizes': crossover_sizes
    }