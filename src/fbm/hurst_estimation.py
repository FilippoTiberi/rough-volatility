import numpy as np
from numpy.typing import ArrayLike
from typing import Sequence


def hurst_variance_scaling(
    series: ArrayLike,
    scales: Sequence[int] | None = None,
) -> float:
    """
    Estimate the Hurst exponent H using variance scaling:

        Var(X_{t+m} - X_t) ~ m^{2H}

    We compute log Var vs log m and estimate the slope via linear regression.
    H ≈ slope / 2.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)

    if scales is None:
        # powers of 2 between 4 and n/4
        max_scale = n // 4
        scales = [m for m in (2 ** np.arange(2, 10)) if m < max_scale]
        if len(scales) < 3:
            scales = [2, 4, 8]

    log_m = []
    log_var = []

    for m in scales:
        diffs = x[m:] - x[:-m]
        v = np.var(diffs)
        if v <= 0:
            continue
        log_m.append(np.log(m))
        log_var.append(np.log(v))

    log_m = np.array(log_m)
    log_var = np.array(log_var)

    A = np.vstack([log_m, np.ones_like(log_m)]).T
    slope, _ = np.linalg.lstsq(A, log_var, rcond=None)[0]
    H = slope / 2.0
    return float(H)
