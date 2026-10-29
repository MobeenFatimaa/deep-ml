import numpy as np

def hybrid_vit_multitask(image, patch_conv_w, patch_conv_b, pos_emb,
                         attn_Wq, attn_Wk, attn_Wv, attn_Wo,
                         num_heads, task_heads):
    """
    Hybrid Vision Transformer for multi-task driving perception.

    Args:
        image: numpy array (H, W, C)
        patch_conv_w: numpy array (D, kH, kW, C)
        patch_conv_b: numpy array (D,)
        pos_emb: numpy array (N, D)
        attn_Wq: numpy array (D, D)
        attn_Wk: numpy array (D, D)
        attn_Wv: numpy array (D, D)
        attn_Wo: numpy array (D, D)
        num_heads: int
        task_heads: dict {name: (W, b)}

    Returns:
        dict {name: numpy array (N, out_dim)}
    """
    D, kH, kW, C = patch_conv_w.shape
    H, W, _ = image.shape
    
    # 1. Convolutional Patch Embedding (non-overlapping patches, stride = kernel size)
    n_h = H // kH
    n_w = W // kW
    patches = []
    for i in range(n_h):
        for j in range(n_w):
            patch = image[i*kH:(i+1)*kH, j*kW:(j+1)*kW, :]
            patches.append(patch)
    patches = np.array(patches) # Shape: (N, kH, kW, C)
    
    # Compute convolution: output for each patch is sum over kH, kW, C of patch * w + b
    # patch_conv_w shape: (D, kH, kW, C)
    # patches shape: (N, kH, kW, C)
    # We want result of shape (N, D)
    # Using einsum: N: batch/patch, D: output channels, h, w, c: spatial/channel
    token_embeds = np.einsum('nhwc,dhwc->nd', patches, patch_conv_w) + patch_conv_b
    
    # 2. Positional Embedding
    x = token_embeds + pos_emb
    
    # 3. Multi-Head Self-Attention with Residual
    N = x.shape[0]
    
    # Project to Q, K, V
    Q = x @ attn_Wq # (N, D)
    K = x @ attn_Wk # (N, D)
    V = x @ attn_Wv # (N, D)
    
    head_dim = D // num_heads
    
    # Reshape for multi-head attention: (N, num_heads, head_dim) -> transpose to (num_heads, N, head_dim)
    Q = Q.reshape(N, num_heads, head_dim).transpose(1, 0, 2)
    K = K.reshape(N, num_heads, head_dim).transpose(1, 0, 2)
    V = V.reshape(N, num_heads, head_dim).transpose(1, 0, 2)
    
    # Scaled dot-product attention per head
    scale = 1.0 / np.sqrt(head_dim)
    scores = np.matmul(Q, K.transpose(0, 2, 1)) * scale # (num_heads, N, N)
    
    # Numerically stable softmax
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # Attention output per head
    attn_out = np.matmul(attn_weights, V) # (num_heads, N, head_dim)
    
    # Concatenate heads back to (N, D)
    attn_out = attn_out.transpose(1, 0, 2).reshape(N, D)
    
    # Output projection
    proj_out = attn_out @ attn_Wo
    
    # Residual connection from pre-attention tokens (x)
    features = proj_out + x
    
    # 4. Task-Specific Prediction Heads
    results = {}
    for task_name, (W_task, b_task) in task_heads.items():
        results[task_name] = features @ W_task + b_task
        
    return results