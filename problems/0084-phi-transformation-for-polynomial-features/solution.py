import numpy as np

def phi_transform(data: list[float], degree: int) -> list[list[float]]:
    """
    Perform a Phi Transformation to map input features into a higher-dimensional space by generating polynomial features.

    Args:
        data (list[float]): A list of numerical values to transform.
        degree (int): The degree of the polynomial expansion.

    Returns:
        list[list[float]]: A nested list where each inner list represents the polynomial features.
    """
    if degree < 0:
        return []
    
    transformed_data = []
    for x in data:
        # Generate powers from 0 to degree for each data point
        row = [x ** d for d in range(degree + 1)]
        transformed_data.append(row)
        
    return transformed_data