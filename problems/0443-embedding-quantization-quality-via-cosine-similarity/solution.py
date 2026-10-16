import numpy as np

def embedding_quantization_quality(embeddings: np.ndarray, num_bits: int = 8) -> dict:
    """
    Measure the quality of uniform embedding quantization using cosine similarity.
    
    Args:
        embeddings: Input embeddings of shape (n, d)
        num_bits: Number of bits for quantization
    
    Returns:
        Dictionary with mean_cosine_similarity, min_cosine_similarity,
        and max_cosine_similarity (each rounded to 6 decimal places)
    """
    embeddings_arr = np.asarray(embeddings, dtype=np.float64)
    n, d = embeddings_arr.shape
    
    num_levels = (1 << num_bits) - 1
    
    cosine_similarities = []
    
    for i in range(n):
        v = embeddings_arr[i]
        
        # Check for zero vectors
        if np.all(v == 0.0):
            cosine_similarities.append(0.0)
            continue
            
        vmin = np.min(v)
        vmax = np.max(v)
        
        # Check for constant vectors (nonzero)
        if vmin == vmax:
            cosine_similarities.append(1.0)
            continue
            
        # Per-vector uniform quantization range
        scale = (vmax - vmin) / num_levels
        
        # Quantize
        q = np.clip(np.round((v - vmin) / scale), 0, num_levels)
        
        # Dequantize
        v_dequant = q * scale + vmin
        
        # Compute cosine similarity
        dot_prod = np.dot(v, v_dequant)
        norm_v = np.linalg.norm(v)
        norm_dequant = np.linalg.norm(v_dequant)
        
        if norm_v == 0.0 or norm_dequant == 0.0:
            sim = 0.0
        else:
            sim = dot_prod / (norm_v * norm_dequant)
            
        cosine_similarities.append(float(sim))
        
    cosine_similarities = np.array(cosine_similarities)
    
    return {
        'mean_cosine_similarity': round(float(np.mean(cosine_similarities)), 6),
        'min_cosine_similarity': round(float(np.min(cosine_similarities)), 6),
        'max_cosine_similarity': round(float(np.max(cosine_similarities)), 6)
    }