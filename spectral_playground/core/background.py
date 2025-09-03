from __future__ import annotations

from typing import Optional

import numpy as np


Array = np.ndarray


class BackgroundModel:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def sample(self, L: int, H: int, W: int, kind: str = "constant", **p) -> Array:
        """Return background B of shape (L, P) with P=H*W.

        kinds:
          - 'constant': per-channel constant offset
          - 'lowrank': rank-r smooth fields via random factors
        """
        if kind == "constant":
            level = float(p.get("level", 0.0))
            B = np.full((L, H * W), level, dtype=np.float32)
            return B

        if kind == "lowrank":
            rank = int(p.get("rank", 2))
            # Random low-rank factors with smoothness via Gaussian blur approximation
            U = self.rng.random((L, rank), dtype=float).astype(np.float32)
            V = self.rng.random((rank, H * W), dtype=float).astype(np.float32)
            B = (U @ V).astype(np.float32)
            return B

        raise ValueError("Unsupported background kind in MVP")


