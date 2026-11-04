import numpy as np

def cosine_similarity(v1, v2):
    """
    Calculate the cosine_similarity of two vectors.
    Args:
        v1 (numpy.ndarray): 1D array representing the first vector.
        v2 (numpy.ndarray): 1D array representing the second vector.
    Returns:
        The cosine_similarity of the two vectors.
    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    
    # Compute the dot product of v1 and v2
    dot_product = np.dot(v1, v2)
    
    # Compute the Euclidean norms (magnitudes) of v1 and v2
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Compute cosine similarity
    similarity = dot_product / (norm_v1 * norm_v2)
    
    return float(similarity)