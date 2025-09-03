from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np


Array = np.ndarray


class PoissonEMUnmixer:
    name = "em_poisson"
    supports_blind = False

    def __init__(self, n_iter: int = 200):
        self.n_iter = int(n_iter)
        self._M: Optional[Array] = None

    def fit(self, Y: Array, *, M: Optional[Array] = None, priors: Optional[dict] = None, **kwargs) -> None:
        if M is None:
            raise ValueError("Poisson EM requires known M for MVP")
        self._M = M.astype(np.float32)

    def transform(self, Y: Array, *, M: Optional[Array] = None) -> Dict[str, Any]:
        M_use = self._M if self._M is not None else M
        if M_use is None:
            raise ValueError("Poisson EM requires known M")
        L, P = Y.shape
        Lm, K = M_use.shape
        if Lm != L:
            raise ValueError("Shape mismatch between Y and M")
        # Initialize A
        rng = np.random.default_rng(0)
        A = np.maximum(rng.random((K, P), dtype=float).astype(np.float32), 1e-6)
        M = M_use.astype(np.float32)
        for _ in range(self.n_iter):
            MA = M @ A + 1e-6
            ratio = Y / MA
            A *= (M.T @ ratio)
            # Normalize step to prevent blow-up
            A = np.maximum(A, 1e-12)
            A /= (1e-6 + np.max(A, axis=0, keepdims=True))
        return {"A": A}


