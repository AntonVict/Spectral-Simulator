from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


Array = np.ndarray


@dataclass
class FieldSpec:
    shape: Tuple[int, int]  # (H, W)
    pixel_size_nm: float


class AbundanceField:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def sample(self, K: int, field: FieldSpec, kind: str = "dots", **kwargs) -> Array:
        """Return A of shape (K, P).

        kinds:
          - 'dots': sparse Gaussian spots per fluor
          - 'uniform': uniform random non-negative
        """
        H, W = field.shape
        P = H * W
        if kind == "uniform":
            A = self.rng.random((K, P), dtype=float)
            return A.astype(np.float32)
        if kind != "dots":
            raise ValueError("Only 'dots' and 'uniform' kinds are supported in MVP")

        density = float(kwargs.get("density_per_100x100_um2", 50.0))
        spot_profile = kwargs.get("spot_profile", {"kind": "gaussian", "sigma_px": 1.2})
        sigma_px = float(spot_profile.get("sigma_px", 1.2))

        # Expected number of spots per 100x100 um^2
        pixel_size_um = field.pixel_size_nm / 1000.0
        area_um2 = (H * pixel_size_um) * (W * pixel_size_um)
        expected_spots = density * (area_um2 / 1.0e4)

        A_maps = np.zeros((K, H, W), dtype=np.float32)
        for k in range(K):
            n_spots = self.rng.poisson(lam=max(expected_spots, 0.0))
            if n_spots == 0:
                continue
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            for _ in range(int(n_spots)):
                cy = self.rng.integers(0, H)
                cx = self.rng.integers(0, W)
                amp = float(self.rng.random()) + 0.5  # avoid too small amplitudes
                g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma_px ** 2))
                A_maps[k] += (amp * g).astype(np.float32)
        A = A_maps.reshape(K, P)
        return A


class PSF:
    """Placeholder PSF for future extensions."""

    def __init__(self, sigma_px: float = 1.0):
        self.sigma_px = float(sigma_px)

    def kernel(self, field: FieldSpec) -> Array:
        H, W = field.shape
        size = int(6 * self.sigma_px + 1)
        size = max(3, size | 1)  # odd size
        c = size // 2
        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        g = np.exp(-(((yy - c) ** 2 + (xx - c) ** 2) / (2.0 * self.sigma_px ** 2)))
        g = g / np.sum(g)
        return g.astype(np.float32)


