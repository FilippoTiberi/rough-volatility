import numpy as np
from typing import Tuple

from .rbergomi_sde import RBergomiModel, RBergomiParams


def price_call_rbergomi_mc(
    S0: float,
    K: float,
    T: float,
    n_steps: int,
    n_paths: int,
    params: RBergomiParams,
    random_state: int | None = None,
) -> Tuple[float, float]:
    """
    Monte Carlo price of a European call under the rBergomi model.

    Returns
    -------
    price : float
    stderr: float
    """
    model = RBergomiModel(params)
    t_grid, S, v = model.simulate_paths(
        S0=S0,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        random_state=random_state,
    )

    ST = S[:, -1]
    payoff = np.maximum(ST - K, 0.0)
    disc_factor = np.exp(-params.r * T)

    price = disc_factor * np.mean(payoff)
    stderr = disc_factor * np.std(payoff, ddof=1) / np.sqrt(n_paths)
    return float(price), float(stderr)
