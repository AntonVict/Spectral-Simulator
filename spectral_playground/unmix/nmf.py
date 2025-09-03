from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np


Array = np.ndarray


class NMFUnmixer:
    name = "nmf"
    supports_blind = True

    def __init__(self, n_iter: int = 200, beta_div: float = 2.0, K: Optional[int] = None, random_state: Optional[int] = None):
        self.n_iter = int(n_iter)
        self.beta_div = float(beta_div)
        self.K = K
        self.random_state = random_state
        self._A: Optional[Array] = None
        self._M: Optional[Array] = None

    def fit(self, Y: Array, *, M: Optional[Array] = None, priors: Optional[dict] = None, **kwargs) -> None:
        rng = np.random.default_rng(self.random_state)
        L, P = Y.shape
        if M is not None:
            # Semi-NMF where M is fixed; only estimate A
            self._M = M.astype(np.float32)
            K = M.shape[1]
            A = rng.random((K, P), dtype=float).astype(np.float32)
            A = np.maximum(A, 1e-6)
            for _ in range(self.n_iter):
                numer = self._M.T @ Y
                denom = (self._M.T @ self._M) @ A + 1e-8
                A *= numer / denom
            self._A = A
        else:
            if self.K is None:
                raise ValueError("When M is unknown, NMF requires K to be specified")
            K = int(self.K)
            W = np.maximum(rng.random((L, K), dtype=float).astype(np.float32), 1e-6)
            H = np.maximum(rng.random((K, P), dtype=float).astype(np.float32), 1e-6)
            for _ in range(self.n_iter):
                # Euclidean beta=2 multiplicative updates
                WH = W @ H
                W *= (Y @ H.T) / (WH @ H.T + 1e-8)
                WH = W @ H
                H *= (W.T @ Y) / (W.T @ WH + 1e-8)
            self._M = W
            self._A = H

    def transform(self, Y: Array, *, M: Optional[Array] = None) -> Dict[str, Any]:
        return {"A": self._A, "M": self._M}


