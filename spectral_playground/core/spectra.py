from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy.stats import norm


Array = np.ndarray


@dataclass
class Fluorophore:
    name: str
    model: Literal["gaussian", "lognormal", "skewnorm", "mixture", "empirical", "weibull"]
    params: dict
    brightness: float = 1.0


@dataclass
class Channel:
    name: str
    center_nm: float
    bandwidth_nm: float
    response: Optional[Array] = None  # sampled r_ell(lambda) over grid


@dataclass
class SpectralSystem:
    lambdas: Array  # wavelength grid (nm), shape (G,)
    channels: list[Channel]
    fluors: list[Fluorophore]

    def _pdf(self, fluor: Fluorophore) -> Array:
        lam = self.lambdas
        model = fluor.model.lower()
        if model == "gaussian":
            mu = float(fluor.params["mu"])  # in nm
            sigma = float(fluor.params["sigma"])  # in nm
            x = (lam - mu) / max(sigma, 1e-12)
            pdf = np.exp(-0.5 * x * x)
        elif model == "lognormal":
            # Parameters are log-space mean (mu) and std (sigma)
            mu = float(fluor.params["mu"])  # log-scale mean
            sigma = float(fluor.params["sigma"])  # log-scale std
            lam_pos = np.clip(lam, 1e-6, None)
            coeff = 1.0 / (lam_pos * sigma * np.sqrt(2.0 * np.pi))
            pdf = coeff * np.exp(-0.5 * ((np.log(lam_pos) - mu) / sigma) ** 2)
        elif model == "skewnorm":
            mu = float(fluor.params["mu"])  # nm
            sigma = float(fluor.params["sigma"])  # nm
            alpha = float(fluor.params.get("alpha", 0.0))
            z = (lam - mu) / max(sigma, 1e-12)
            pdf = 2.0 * norm.pdf(z) * norm.cdf(alpha * z)
        elif model == "mixture":
            mus = np.asarray(fluor.params["mus"], dtype=float)
            sigmas = np.asarray(fluor.params["sigmas"], dtype=float)
            weights = np.asarray(fluor.params.get("weights", np.ones_like(mus)), dtype=float)
            weights = weights / np.sum(weights)
            pdf = np.zeros_like(lam, dtype=float)
            for m, s, w in zip(mus, sigmas, weights):
                z = (lam - m) / max(float(s), 1e-12)
                pdf += w * np.exp(-0.5 * z * z)
        elif model == "empirical":
            # Support two formats:
            # 1. Raw CSV data (csv_wavelengths + csv_intensities) - interpolates to current grid
            # 2. Pre-interpolated samples (legacy) - must match grid exactly
            
            if "csv_wavelengths" in fluor.params and "csv_intensities" in fluor.params:
                # Format 1: Raw CSV data - interpolate to current grid
                csv_wl = np.asarray(fluor.params["csv_wavelengths"], dtype=float)
                csv_int = np.asarray(fluor.params["csv_intensities"], dtype=float)
                
                # Interpolate to current wavelength grid
                pdf = np.interp(self.lambdas, csv_wl, csv_int, left=0.0, right=0.0)
                pdf = np.clip(pdf, 0.0, None)
            elif "samples" in fluor.params:
                # Format 2: Pre-interpolated samples (must match current grid)
                data = fluor.params.get("samples")
                if data is None:
                    raise ValueError("empirical model requires 'samples' array over lambdas grid")
                arr = np.asarray(data, dtype=float)
                if arr.shape != self.lambdas.shape:
                    raise ValueError(f"empirical samples shape {arr.shape} must match lambdas shape {self.lambdas.shape}")
                pdf = np.clip(arr, 0.0, None)
            else:
                raise ValueError("empirical model requires either 'csv_wavelengths'+'csv_intensities' or 'samples'")
        elif model == "weibull":
            # Params: k (shape), lam (scale), shift (optional nm offset)
            k = float(fluor.params.get("k", 2.0))
            lam = float(fluor.params.get("lam", 20.0))
            shift = float(fluor.params.get("shift", 0.0))
            x = np.clip(self.lambdas - shift, 1e-6, None)
            pdf = (k / lam) * (x / lam) ** (k - 1.0) * np.exp(- (x / lam) ** k)
        else:
            raise ValueError(f"Unknown spectral model: {model}")

        # Normalize to unit area on the discrete grid
        step = float(self._grid_step())
        area = np.sum(pdf) * step
        if area <= 0:
            raise ValueError("Spectral PDF has non-positive area after evaluation")
        pdf = pdf / area
        # Apply brightness as a scale to the resulting column later (handled in build_M)
        return pdf

    def _grid_step(self) -> float:
        if self.lambdas.size < 2:
            return 1.0
        return float(self.lambdas[1] - self.lambdas[0])

    def _channel_response(self, ch: Channel) -> Array:
        if ch.response is not None:
            resp = np.asarray(ch.response, dtype=float)
            if resp.shape != self.lambdas.shape:
                raise ValueError("Channel.response must match lambdas shape")
            return np.clip(resp, 0.0, 1.0)
        # Default: tophat filter
        half_bw = 0.5 * float(ch.bandwidth_nm)
        low = float(ch.center_nm) - half_bw
        high = float(ch.center_nm) + half_bw
        lam = self.lambdas
        resp = (lam >= low) & (lam <= high)
        return resp.astype(float)

    def build_M(self) -> Array:
        """Build system matrix M (L x K) by numerical quadrature.

        Integrates r_ell(lambda) * s_k(lambda) over grid for each channel ell and fluor k.
        Fluor brightness scales each column.
        """
        L = len(self.channels)
        K = len(self.fluors)
        step = self._grid_step()
        R = np.stack([self._channel_response(ch) for ch in self.channels], axis=0)  # (L, G)
        Scols = [self._pdf(f) for f in self.fluors]  # list of (G,)
        S = np.stack(Scols, axis=1)  # (G, K)
        # Integral approximation: sum_g R[L,g] * S[g,K] * step
        M = R @ (S * step)
        # Apply brightness scaling per fluorophore
        brightness = np.asarray([f.brightness for f in self.fluors], dtype=float)
        M = M * brightness.reshape(1, -1)
        M = np.clip(M, 0.0, None)
        return M.astype(np.float32)


