from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .spectra import SpectralSystem
from .spatial import FieldSpec
from .background import BackgroundModel
from .noise import NoiseModel


Array = np.ndarray


@dataclass
class ForwardConfig:
    use_S: bool = False
    use_PSF: bool = False
    photon_counting: bool = True


class ForwardModel:
    def __init__(
        self,
        spectral: SpectralSystem,
        field: FieldSpec,
        bg: BackgroundModel,
        noise: NoiseModel,
        cfg: ForwardConfig,
    ) -> None:
        self.spectral = spectral
        self.field = field
        self.bg = bg
        self.noise = noise
        self.cfg = cfg
        self.M = spectral.build_M()

    def synthesize(
        self,
        A: Array,
        *,
        B: Optional[Array] = None,
        S: Optional[Array] = None,
        psf: Optional[object] = None,
        noise_params: Optional[dict] = None,
    ) -> Array:
        """Return Y of shape (L, P). Handles MVP forward model Y = M A + B + N.

        This MVP ignores S and PSF even if provided; placeholders for future phases.
        """
        L = self.M.shape[0]
        H, W = self.field.shape
        P = H * W
        if A.shape != (self.M.shape[1], P):
            raise ValueError("A must have shape (K, P)")

        if B is None:
            B = self.bg.sample(L, H, W, kind="constant", level=0.0)

        Y_clean = (self.M @ A).astype(np.float32) + B.astype(np.float32)

        nparams = dict(kind="poisson_gaussian", gain=1.0, read_sigma=0.0, dark_rate=0.0)
        if noise_params:
            nparams.update(noise_params)
        Y = self.noise.apply(Y_clean, **nparams)
        return Y.astype(np.float32)


