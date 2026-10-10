import numpy as np

def ring_attention_simulate(Q: np.ndarray, K: np.ndarray, V: np.ndarray, num_devices: int) -> tuple:
    """
    Simulate Ring Attention for Context Parallelism.
    
    Args:
        Q: Query matrix of shape (seq_len, d)
        K: Key matrix of shape (seq_len, d)
        V: Value matrix of shape (seq_len, d)
        num_devices: Number of devices in the ring
    
    Returns:
        Tuple of:
        - output: Attention output of shape (seq_len, d), rounded to 4 decimals
        - comm_schedule: List of lists showing KV source device per step
    """
    seq_len, d = Q.shape
    chunk_size = seq_len // num_devices
    
    # Split Q, K, V into chunks per device
    Q_chunks = [Q[i * chunk_size : (i + 1) * chunk_size] for i in range(num_devices)]
    K_chunks = [K[i * chunk_size : (i + 1) * chunk_size] for i in range(num_devices)]
    V_chunks = [V[i * chunk_size : (i + 1) * chunk_size] for i in range(num_devices)]
    
    scale = 1.0 / np.sqrt(d)
    
    # Track online softmax state for each device's query chunk:
    # m_i: running max score per query row
    # l_i: running sum of exponentials per query row
    # acc_o: running weighted sum of values per query row
    device_states = []
    for i in range(num_devices):
        c_len = Q_chunks[i].shape[0]
        device_states.append({
            'm': np.full((c_len, 1), -np.inf),
            'l': np.zeros((c_len, 1)),
            'acc_o': np.zeros((c_len, d))
        })
        
    comm_schedule = []
    
    # Ring communication steps
    for step in range(num_devices):
        step_sources = []
        next_K_chunks = [None] * num_devices
        next_V_chunks = [None] * num_devices
        
        for dev_id in range(num_devices):
            # In ring attention at step 'step', device i holds the KV chunk from source device:
            # source_dev = (dev_id - step) % num_devices
            source_dev = (dev_id - step) % num_devices
            step_sources.append(source_dev)
            
            curr_K = K_chunks[source_dev]
            curr_V = V_chunks[source_dev]
            
            # Compute attention scores between local query chunk and current KV chunk
            scores = np.dot(Q_chunks[dev_id], curr_K.T) * scale  # shape (chunk_size, chunk_size_kv)
            
            # Online softmax update
            state = device_states[dev_id]
            m_curr = state['m']
            l_curr = state['l']
            acc_o = state['acc_o']
            
            block_max = np.max(scores, axis=-1, keepdims=True)
            new_m = np.maximum(m_curr, block_max)
            
            # Rescale previous accumulation and denominator
            exp_old = np.exp(m_curr - new_m)
            exp_curr = np.exp(scores - new_m)
            
            new_l = exp_old * l_curr + np.sum(exp_curr, axis=-1, keepdims=True)
            new_acc_o = exp_old * acc_o + np.dot(exp_curr, curr_V)
            
            state['m'] = new_m
            state['l'] = new_l
            state['acc_o'] = new_acc_o
            
        comm_schedule.append(step_sources)
        
    # Finalize outputs across devices
    final_output_chunks = []
    for dev_id in range(num_devices):
        state = device_states[dev_id]
        out_chunk = state['acc_o'] / state['l']
        final_output_chunks.append(out_chunk)
        
    output = np.vstack(final_output_chunks)
    output = np.round(output, 4)
    
    return output, comm_schedule