def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    if not vectors or not vectors[0]:
        return []
    
    num_features = len(vectors)
    num_observations = len(vectors[0])
    
    # Sample covariance requires at least 2 observations (division by N - 1)
    if num_observations < 2:
        return []
    
    # Step 1: Calculate the mean for each feature vector
    means = [sum(feature) / num_observations for feature in vectors]
    
    # Step 2: Initialize an m x m matrix with zeros
    cov_matrix = [[0.0 for _ in range(num_features)] for _ in range(num_features)]
    
    # Step 3: Compute sample covariance for each pair of features (i, j)
    for i in range(num_features):
        for j in range(i, num_features):
            # Sum of products of deviations
            sum_dev = sum(
                (vectors[i][k] - means[i]) * (vectors[j][k] - means[j])
                for k in range(num_observations)
            )
            
            cov_value = sum_dev / (num_observations - 1)
            
            # Covariance matrix is symmetric: Cov(X, Y) = Cov(Y, X)
            cov_matrix[i][j] = cov_value
            cov_matrix[j][i] = cov_value
            
    return cov_matrix