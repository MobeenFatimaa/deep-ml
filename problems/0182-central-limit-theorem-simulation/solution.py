import numpy as np

def simulate_clt(distribution: str, n: int, runs: int = 10000, seed: int = 42) -> dict:
    """
    Simulate the Central Limit Theorem.

    Args:
        distribution (str): The distribution to sample from ('uniform', 'exponential', 'bernoulli').
        n (int): Sample size.
        runs (int): Number of repeated experiments.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: {'mean': float, 'std': float} of the standardized sample means.
    """
    np.random.seed(seed)
    
    # 1. Draw samples and set known population parameters (mu, sigma)
    if distribution == 'uniform':
        samples = np.random.uniform(0, 1, size=(runs, n))
        mu = 0.5
        sigma = np.sqrt(1.0 / 12.0)
    elif distribution == 'exponential':
        samples = np.random.exponential(1.0, size=(runs, n))
        mu = 1.0
        sigma = 1.0
    elif distribution == 'bernoulli':
        samples = (np.random.rand(runs, n) < 0.3).astype(float)
        mu = 0.3
        sigma = np.sqrt(0.3 * 0.7)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")
        
    # 2. Compute sample means for each run (axis=1)
    sample_means = np.mean(samples, axis=1)
    
    # 3. Standardize the sample means to Z-scores using the standard error (sigma / sqrt(n))
    std_error = sigma / np.sqrt(n)
    z_scores = (sample_means - mu) / std_error
    
    # 4. Compute population mean and standard deviation (ddof=0) of the Z-scores
    mean_val = float(np.mean(z_scores))
    std_val = float(np.std(z_scores, ddof=0))
    
    return {
        'mean': round(mean_val, 3),
        'std': round(std_val, 3)
    }