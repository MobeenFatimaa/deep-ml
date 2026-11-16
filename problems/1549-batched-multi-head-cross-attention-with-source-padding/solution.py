import numpy as np


def multi_head_cross_attention(decoder_x, encoder_out, W_q, W_k, W_v, W_o, num_heads, src_pad_mask=None):
    """
    Batched multi-head encoder-decoder attention.

    decoder_x:    (B, T, d_model)  queries come from here
    encoder_out:  (B, S, d_model)  keys and values come from here
    W_q, W_k, W_v, W_o: (d_model, d_model)
    num_heads:    H, with d_model divisible by H
    src_pad_mask: optional (B, S) boolean / 0-1 array, True or 1 at padded source positions

    Returns (output, weights) with shapes (B, T, d_model) and (B, H, T, S).
    Masked source positions must have weight exactly 0 for every head and target position.
    """
    B, T, d_model = decoder_x.shape
    S = encoder_out.shape[1]
    H = num_heads
    d_k = d_model // H

    # 1. Linear projections
    Q = np.dot(decoder_x, W_q)       # (B, T, d_model)
    K = np.dot(encoder_out, W_k)     # (B, S, d_model)
    V = np.dot(encoder_out, W_v)     # (B, S, d_model)

    # 2. Reshape into multi-head format and transpose to (B, H, length, d_k)
    Q = Q.reshape(B, T, H, d_k).transpose(0, 2, 1, 3)
    K = K.reshape(B, S, H, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, H, d_k).transpose(0, 2, 1, 3)

    # 3. Scaled dot-product attention scores: (B, H, T, S)
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)

    # 4. Handle source padding mask
    if src_pad_mask is not None:
        mask = src_pad_mask.astype(bool)
        # Broadcastable mask shape: (B, 1, 1, S)
        mask_expanded = mask[:, None, None, :]
        # Set padded scores to a very large negative number so softmax approaches zero
        scores = np.where(mask_expanded, -1e9, scores)

    # 5. Softmax over the source axis (axis=-1) with numerical stability
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    
    if src_pad_mask is not None:
        # Force weights for padded source positions to be exactly 0.0
        exp_scores = np.where(mask_expanded, 0.0, exp_scores)
        
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    weights = exp_scores / sum_exp

    # 6. Compute attention context vector: (B, H, T, d_k)
    context = np.matmul(weights, V)

    # 7. Concatenate heads and project through W_o
    context = context.transpose(0, 2, 1, 3).reshape(B, T, d_model)
    output = np.dot(context, W_o)

    return output, weights