import numpy as np

def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
    B_matrix = np.array(B, dtype=float)
    C_matrix = np.array(C, dtype=float)

    P = np.linalg.inv(C_matrix) @ B_matrix

    return P.tolist()