"""2PL Item Response Theory calibration over model response data.

Joint MLE of model abilities (theta) and item parameters (a, b) under
P(correct | theta, a, b) = 1 / (1 + exp(-a (theta - b))).

We use scipy.optimize.minimize on the negative log-likelihood with a
standard-normal prior on theta to identify the scale.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

logger = logging.getLogger(__name__)


class IRTCalibrator:
    def __init__(self, max_iter: int = 200, tol: float = 1e-5):
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, responses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fit 2PL parameters.

        Args:
            responses: (n_models, n_items) binary correctness matrix.

        Returns:
            (a, b) arrays of length n_items: discrimination and difficulty.
        """
        if responses.ndim != 2:
            raise ValueError("responses must be 2D (n_models, n_items)")
        n_models, n_items = responses.shape
        if n_models < 2 or n_items < 1:
            return np.ones(n_items), np.zeros(n_items)

        # Parameter packing: [theta(n_models), a(n_items), b(n_items)].
        # theta ~ N(0,1) prior; a ~ logN(0, 0.5) (positivity); b unconstrained.
        def unpack(x):
            theta = x[:n_models]
            a = np.exp(x[n_models:n_models + n_items])
            b = x[n_models + n_items:]
            return theta, a, b

        def neg_log_lik(x):
            theta, a, b = unpack(x)
            z = a[None, :] * (theta[:, None] - b[None, :])
            p = expit(z)
            eps = 1e-9
            ll = (responses * np.log(p + eps) + (1 - responses) * np.log(1 - p + eps)).sum()
            # Priors.
            ll -= 0.5 * np.sum(theta ** 2)
            # log a ~ N(0, 0.5) -> penalty 2*(log a)^2.
            ll -= 2.0 * np.sum(x[n_models:n_models + n_items] ** 2)
            return -ll

        x0 = np.concatenate([
            np.zeros(n_models),
            np.zeros(n_items),                  # log a = 0 => a=1
            (0.5 - responses.mean(axis=0)) * 2,  # initial difficulty
        ])
        res = minimize(neg_log_lik, x0, method="L-BFGS-B",
                       options={"maxiter": self.max_iter, "gtol": self.tol})
        if not res.success:
            logger.warning("IRT fit did not fully converge: %s", res.message)
        _, a, b = unpack(res.x)
        return a, b

    def flag_items(
        self,
        a: np.ndarray,
        b: np.ndarray,
        floor_threshold: float = -2.5,
        ceiling_threshold: float = 2.5,
        low_disc_threshold: float = 0.5,
    ) -> dict[str, list[int]]:
        floor = [int(i) for i, v in enumerate(b) if v < floor_threshold]
        ceiling = [int(i) for i, v in enumerate(b) if v > ceiling_threshold]
        low_disc = [int(i) for i, v in enumerate(a) if v < low_disc_threshold]
        return {"floor": floor, "ceiling": ceiling, "low_disc": low_disc}
