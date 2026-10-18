import numpy as np


def deterministic_hash(s):
    """Converts a string to a deterministic integer."""
    h = 0
    for c in str(s):
        h = (h * 31 + ord(c)) % (2**31)
    return h


def create_hv(dim, seed):
    """Creates a bipolar hypervector of given dimension using the seed."""
    np.random.seed(seed % (2**32 - 1))
    return np.random.choice([-1, 1], dim)


def create_row_hv(row, dim, random_seeds):
    """Create composite hypervector for a dataset row.

    Hint: For each feature, the value seed should combine the base seed
    with the hashed value using modular arithmetic.
    """
    bound_vectors = []

    for name, value in row.items():
        base = random_seeds[name]
        
        # Name vector
        name_hv = create_hv(dim, base)
        
        # Value vector
        value_seed = (base + deterministic_hash(value)) % (2**32 - 1)
        value_hv = create_hv(dim, value_seed)
        
        # Bind the pair by element-wise multiplication
        bound = name_hv * value_hv
        bound_vectors.append(bound)

    # Bundle: sum all bound vectors element-wise
    bundled_sum = np.sum(bound_vectors, axis=0)

    # Map each entry to 1 if >= 0, and -1 otherwise
    row_hv = np.where(bundled_sum >= 0, 1, -1)

    return row_hv


# Example verification
if __name__ == "__main__":
    row = {"FeatureA": "value1", "FeatureB": "value2"}
    dim = 5
    random_seeds = {"FeatureA": 42, "FeatureB": 7}
    print(create_row_hv(row, dim, random_seeds))