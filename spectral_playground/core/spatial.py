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
          - 'circles': filled disks per fluor
          - 'boxes': filled squares per fluor
          - 'gaussian_blobs': Gaussian blobs with configurable sigma
          - 'mixed': mixture of circles/boxes/gaussian_blobs per fluor
        """
        H, W = field.shape
        P = H * W
        if kind == "uniform":
            A = self.rng.random((K, P), dtype=float)
            return A.astype(np.float32)
        
        # Initialize output maps
        A_maps = np.zeros((K, H, W), dtype=np.float32)
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

        if kind == "dots":
            density = float(kwargs.get("density_per_100x100_um2", 50.0))
            spot_profile = kwargs.get("spot_profile", {"kind": "gaussian", "sigma_px": 1.2})
            sigma_px = float(spot_profile.get("sigma_px", 1.2))

            # Expected number of spots per 100x100 um^2
            pixel_size_um = field.pixel_size_nm / 1000.0
            area_um2 = (H * pixel_size_um) * (W * pixel_size_um)
            expected_spots = density * (area_um2 / 1.0e4)

            for k in range(K):
                n_spots = self.rng.poisson(lam=max(expected_spots, 0.0))
                if n_spots == 0:
                    continue
                for _ in range(int(n_spots)):
                    cy = int(self.rng.integers(0, H))
                    cx = int(self.rng.integers(0, W))
                    amp = float(self.rng.random()) + 0.5  # avoid too small amplitudes
                    g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma_px ** 2))
                    A_maps[k] += (amp * g).astype(np.float32)
            return A_maps.reshape(K, P)

        # New object-based generators
        count_per_fluor = int(kwargs.get("count_per_fluor", 50))
        size_px = float(kwargs.get("size_px", 6.0))
        intensity_min = float(kwargs.get("intensity_min", 0.5))
        intensity_max = float(kwargs.get("intensity_max", 1.5))

        def add_circle(map_ref: Array, cy: int, cx: int, radius: float, amplitude: float) -> None:
            rr2 = (yy - cy) ** 2 + (xx - cx) ** 2
            mask = rr2 <= (radius ** 2)
            map_ref[mask] += amplitude

        def add_box(map_ref: Array, cy: int, cx: int, half: int, amplitude: float) -> None:
            y0 = max(0, cy - half)
            y1 = min(H, cy + half + 1)
            x0 = max(0, cx - half)
            x1 = min(W, cx + half + 1)
            map_ref[y0:y1, x0:x1] += amplitude

        def add_gaussian(map_ref: Array, cy: int, cx: int, sigma: float, amplitude: float) -> None:
            g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2))
            map_ref += amplitude * g

        def rand_amp() -> float:
            return float(self.rng.uniform(intensity_min, intensity_max))

        all_types = ("circles", "boxes", "gaussian_blobs")

        for k in range(K):
            n = count_per_fluor
            for _ in range(max(0, n)):
                cy = int(self.rng.integers(0, H))
                cx = int(self.rng.integers(0, W))
                amp = rand_amp()
                obj_type = kind
                if kind == "mixed":
                    obj_type = all_types[int(self.rng.integers(0, len(all_types)))]

                if obj_type == "circles":
                    add_circle(A_maps[k], cy, cx, radius=size_px, amplitude=amp)
                elif obj_type == "boxes":
                    add_box(A_maps[k], cy, cx, half=int(max(1, round(size_px))), amplitude=amp)
                elif obj_type == "gaussian_blobs":
                    add_gaussian(A_maps[k], cy, cx, sigma=size_px, amplitude=amp)
                else:
                    raise ValueError(f"Unknown object kind: {obj_type}")

        return A_maps.reshape(K, P)

    def build_from_objects(
        self,
        K: int,
        field: FieldSpec,
        objects: list,
        base: Array | None = None,
    ) -> Array:
        """Build abundance maps from a list of object specs.

        Each object is a dict with keys:
          - 'fluor_index': int (0..K-1)
          - 'kind': 'circles' | 'boxes' | 'gaussian_blobs' | 'dots'
          - 'region': {'type': 'full' | 'rect' | 'circle', ...}
          - 'count': int
          - 'size_px': float
          - 'intensity_min': float
          - 'intensity_max': float
          - 'spot_sigma': float (for 'dots'/'gaussian_blobs')
        """
        H, W = field.shape
        P = H * W
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

        A_maps = np.zeros((K, H, W), dtype=np.float32) if base is None else base.reshape(K, H, W).astype(np.float32)

        def region_mask(region: dict) -> Array:
            rtype = region.get("type", "full")
            if rtype == "full":
                return np.ones((H, W), dtype=bool)
            if rtype == "rect":
                x0 = int(max(0, region.get("x0", 0)))
                y0 = int(max(0, region.get("y0", 0)))
                w = int(max(1, region.get("w", W)))
                h = int(max(1, region.get("h", H)))
                x1 = min(W, x0 + w)
                y1 = min(H, y0 + h)
                mask = np.zeros((H, W), dtype=bool)
                mask[y0:y1, x0:x1] = True
                return mask
            if rtype == "circle":
                cx = float(region.get("cx", W / 2))
                cy = float(region.get("cy", H / 2))
                r = float(region.get("r", min(H, W) / 3))
                return ((yy - cy) ** 2 + (xx - cx) ** 2) <= (r ** 2)
            raise ValueError(f"Unknown region type: {rtype}")

        def add_circle(map_ref: Array, cy: int, cx: int, radius: float, amplitude: float, mask: Array) -> None:
            rr2 = (yy - cy) ** 2 + (xx - cx) ** 2
            local = (rr2 <= (radius ** 2)).astype(np.float32) * amplitude
            map_ref += local * mask

        def add_box(map_ref: Array, cy: int, cx: int, half: int, amplitude: float, mask: Array) -> None:
            y0 = max(0, cy - half)
            y1 = min(H, cy + half + 1)
            x0 = max(0, cx - half)
            x1 = min(W, cx + half + 1)
            local = np.zeros((H, W), dtype=np.float32)
            local[y0:y1, x0:x1] = amplitude
            map_ref += local * mask

        def add_gaussian(map_ref: Array, cy: int, cx: int, sigma: float, amplitude: float, mask: Array) -> None:
            g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
            map_ref += (amplitude * g) * mask

        for obj in (objects or []):
            k = int(obj.get("fluor_index", 0))
            if not (0 <= k < K):
                continue
            kind = obj.get("kind", "gaussian_blobs")
            region = obj.get("region", {"type": "full"})
            cnt = int(obj.get("count", 50))
            size_px = float(obj.get("size_px", 6.0))
            imin = float(obj.get("intensity_min", 0.5))
            imax = float(obj.get("intensity_max", 1.5))
            spot_sigma = float(obj.get("spot_sigma", max(1.0, size_px / 3.0)))

            mask = region_mask(region)

            # Sampling positions across full image; region mask applied to shape
            for _ in range(max(0, cnt)):
                cy = int(self.rng.integers(0, H))
                cx = int(self.rng.integers(0, W))
                amp = float(self.rng.uniform(imin, imax))
                if kind == "circles":
                    add_circle(A_maps[k], cy, cx, radius=size_px, amplitude=amp, mask=mask)
                elif kind == "boxes":
                    add_box(A_maps[k], cy, cx, half=int(max(1, round(size_px))), amplitude=amp, mask=mask)
                elif kind in ("gaussian_blobs", "dots"):
                    add_gaussian(A_maps[k], cy, cx, sigma=spot_sigma, amplitude=amp, mask=mask)
                else:
                    raise ValueError(f"Unknown object kind: {kind}")

        return A_maps.reshape(K, P)


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


