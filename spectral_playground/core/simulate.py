from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .spectra import SpectralSystem
from .spatial import FieldSpec
from .background import BackgroundModel


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
        cfg: ForwardConfig,
    ) -> None:
        self.spectral = spectral
        self.field = field
        self.bg = bg
        self.cfg = cfg
        self.M = spectral.build_M()

    def synthesize(
        self,
        A: Array,
        *,
        B: Optional[Array] = None,
        S: Optional[Array] = None,
        psf: Optional[object] = None,
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

        return Y_clean.astype(np.float32)


