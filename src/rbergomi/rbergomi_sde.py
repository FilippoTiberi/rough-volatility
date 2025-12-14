import numpy as np
from dataclasses import dataclass
from typing import Tuple

from fbm.cholesky_fbm import simulate_fbm_cholesky


@dataclass
class RBergomiParams:
    hurst: float
    eta: float          # vol-of-vol
    rho: float          # correlation between price and vol driver
    xi0: float = 0.04   # initial forward variance level
    r: float = 0.0      # risk-free rate


class RBergomiModel:
    """
    Toy implementation of the rough Bergomi model:

        v_t = xi0 * exp( η B_t^H - 0.5 η^2 t^{2H} )
        dS_t = S_t * sqrt(v_t) dW_t

    We simulate B^H via Cholesky fBm approximation and correlate W and the
    Brownian driver of B^H through rho.
    """

    def __init__(self, params: RBergomiParams):
        self.params = params

    def simulate_paths(
        self,
        S0: float,
        T: float,
        n_steps: int,
        n_paths: int,
        random_state: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate S_t and v_t under rBergomi using a simple Euler scheme.

        Returns
        -------
        t_grid : (n_steps+1,)
        S      : (n_paths, n_steps+1)
        v      : (n_paths, n_steps+1)
        """
        p = self.params
        rng = np.random.default_rng(random_state)
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        # fBm for volatility driver
        t_fbm, fbm_paths = simulate_fbm_cholesky(
            n_steps=n_steps,
            hurst=p.hurst,
            T=T,
            n_paths=n_paths,
            random_state=random_state,
        )

        # Brownian motions with correlation rho
        Z1 = rng.standard_normal(size=(n_paths, n_steps))
        Z2 = rng.standard_normal(size=(n_paths, n_steps))
        dWv = Z1 * sqrt_dt
        dW = (p.rho * Z1 + np.sqrt(1 - p.rho**2) * Z2) * sqrt_dt

        t_grid = np.linspace(0.0, T, n_steps + 1)

        # Variance process
        v = np.zeros((n_paths, n_steps + 1))
        S = np.zeros((n_paths, n_steps + 1))

        v[:, 0] = p.xi0
        S[:, 0] = S0

        # Approximate B^H increments from fbm_paths
        BH = fbm_paths  # (n_paths, n_steps+1)

        for i in range(1, n_steps + 1):
            t = t_grid[i]
            # log variance as in rBergomi (toy)
            exponent = p.eta * BH[:, i] - 0.5 * p.eta**2 * t ** (2 * p.hurst)
            v[:, i] = p.xi0 * np.exp(exponent)

            # Euler for S
            S[:, i] = S[:, i - 1] * np.exp(
                (p.r - 0.5 * v[:, i - 1]) * dt + np.sqrt(v[:, i - 1]) * dW[:, i - 1]
            )

        return t_grid, S, v
