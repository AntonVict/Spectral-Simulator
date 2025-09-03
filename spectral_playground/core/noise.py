from __future__ import annotations

from typing import Optional

import numpy as np


Array = np.ndarray


class NoiseModel:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def apply(
        self,
        Y_clean: Array,
        *,
        kind: str = "poisson_gaussian",
        gain: float = 1.0,
        read_sigma: float = 0.0,
        dark_rate: float = 0.0,
    ) -> Array:
        """Apply noise to clean signal.

        - poisson_gaussian: Poisson(gain * (Y + dark)) + N(0, read_sigma^2)
        """
        if kind != "poisson_gaussian":
            raise ValueError("Only 'poisson_gaussian' is implemented in MVP")
        signal = np.clip(Y_clean, 0.0, None)
        lam = gain * (signal + dark_rate)
        counts = self.rng.poisson(lam).astype(np.float32)
        noisy = counts + self.rng.normal(0.0, read_sigma, size=counts.shape).astype(np.float32)
        # Clip to non-negative
        return np.clip(noisy, 0.0, None)


