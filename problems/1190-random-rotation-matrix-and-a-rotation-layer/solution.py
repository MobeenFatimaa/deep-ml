import numpy as np


def rotation_layer(X, angle):
    """Rotates a list of 2D points by a given angle in radians using a 2x2 rotation

    matrix.

    Args:
        X: List of 2D points [[x1, y1], [x2, y2], ...]
        angle: Rotation angle in radians (float)

    Returns:
        List of rotated 2D points [[x1', y1'], [x2', y2'], ...]
    """
    # Build 2x2 counter-clockwise rotation matrix:
    # R = [[cos(theta), -sin(theta)],
    #      [sin(theta),  cos(theta)]]
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # Convert X to numpy array of shape (N, 2)
    X_arr = np.array(X, dtype=float)

    # Transform points: X_rotated = X @ R.T  (equivalent to R @ x for column vectors)
    X_rotated = np.dot(X_arr, R.T)

    return X_rotated.tolist()