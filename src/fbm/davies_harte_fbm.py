import numpy as np

def _fgn_autocov(h: float, n: int) -> np.ndarray:
    k = np.arange(0, n, dtype=float)
    gamma = 0.5 * (np.abs(k + 1) ** (2 * h)
                   - 2 * np.abs(k) ** (2 * h)
                   + np.abs(k - 1) ** (2 * h))
    return gamma


def simulate_fbm_davies_harte(
    n_steps: int,
    hurst: float,
    T: float = 1.0,
    n_paths: int = 1,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Davies–Harte simulation of fBm via fractional Gaussian noise.

    This is a simple implementation intended for educational use.
    """
    rng = np.random.default_rng(random_state)
    dt = T / n_steps

    # Autocovariance of FGN
    gamma = _fgn_autocov(hurst, n_steps)
    # Build circulant covariance
    first_row = np.concatenate([gamma, gamma[-2:0:-1]])
    lam = np.fft.fft(first_row).real

    if np.any(lam < 0):
        raise ValueError("Davies–Harte eigenvalues negative; try smaller n_steps or different H.")

    lam_sqrt = np.sqrt(lam / (2 * len(first_row)))

    paths = []
    for _ in range(n_paths):
        z = rng.standard_normal(len(first_row)) + 1j * rng.standard_normal(len(first_row))
        w = lam_sqrt * z
        fgn = np.fft.ifft(w).real[:n_steps]
        fgn = fgn * (dt ** hurst)
        fbm = np.concatenate([[0.0], np.cumsum(fgn)])
        paths.append(fbm)

    t_grid = np.linspace(0.0, T, n_steps + 1)
    return t_grid, np.array(paths)
