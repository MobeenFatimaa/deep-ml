import numpy as np


def logit_margin(z, t):
    """Margin m = z_t - log sum_{j != t} exp(z_j), computed stably; returns a Python float."""
    z_arr = np.asarray(z, dtype=np.float64)
    other = np.delete(z_arr, t)
    max_other = np.max(other)
    lse = max_other + np.log(np.sum(np.exp(other - max_other)))
    return float(z_arr[t] - lse)


def precision_cliff(dtype):
    """Smallest m >= 0 at which 1/(1 + exp(-m)) == 1 when evaluated in dtype; bisection on a float64 m.

    Returns the margin rounded to 4 decimals.
    """
    def is_saturated(m):
        em = np.exp(-m)
        denom = dtype(1.0) + dtype(em)
        val = dtype(1.0) / denom
        return val == dtype(1.0)

    low = 0.0
    high = 100.0

    for _ in range(200):
        mid = 0.5 * (low + high)
        if is_saturated(mid):
            high = mid
        else:
            low = mid
        if high - low < 1e-12:
            break

    return round(float(high), 4)