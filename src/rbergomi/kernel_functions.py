import numpy as np

def power_law_kernel(t: float, s: float, hurst: float) -> float:
    """
    Simple power-law kernel K(t,s) = (t - s)^(H - 1/2) for 0 <= s < t.
    """
    if s >= t:
        return 0.0
    return (t - s) ** (hurst - 0.5)


def discrete_kernel_matrix(
    t_grid: np.ndarray,
    hurst: float,
) -> np.ndarray:
    """
    Build a lower-triangular discretised kernel matrix K_{ij} = K(t_i, t_j).

    This can be used in simple Volterra approximations of rough volatility models.
    """
    n = len(t_grid)
    K = np.zeros((n, n))
    for i in range(1, n):
        for j in range(i):
            K[i, j] = power_law_kernel(t_grid[i], t_grid[j], hurst)
    return K
