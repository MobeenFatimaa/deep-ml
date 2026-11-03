import numpy as np

def multi_lora_linear(x, W, A, B, lora_ids, scales):
    """
    Batched linear with a per-row LoRA adapter.

    Returns y of shape (batch, out_features).
    """
    # 1. Base GEMM: computes x @ W for all rows
    y = np.matmul(x, W)
    
    # Convert inputs to numpy arrays
    x = np.asarray(x)
    lora_ids = np.asarray(lora_ids)
    scales = np.asarray(scales)
    
    # 2. Add per-row LoRA contributions where lora_ids[i] >= 0
    for i in range(len(lora_ids)):
        adapter_id = lora_ids[i]
        if adapter_id < 0:
            continue
        
        # Extract specific adapter matrices
        A_id = A[adapter_id]  # shape: (rank, in_features)
        B_id = B[adapter_id]  # shape: (out_features, rank)
        
        # Compute LoRA path: (x @ A.T) @ B.T
        x_A = np.matmul(x[i], A_id.T)
        lora_out = np.matmul(x_A, B_id.T)
        
        # Apply scaling and add to the base output for row i
        y[i] += scales[adapter_id] * lora_out

    return y