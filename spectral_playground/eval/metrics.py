from __future__ import annotations

import numpy as np


Array = np.ndarray


def rmse(a: Array, b: Array) -> float:
    diff = np.asarray(a) - np.asarray(b)
    return float(np.sqrt(np.mean(diff ** 2)))


def mae(a: Array, b: Array) -> float:
    diff = np.asarray(a) - np.asarray(b)
    return float(np.mean(np.abs(diff)))


def sam(a: Array, b: Array, axis: int = 0) -> float:
    """Spectral Angle Mapper in radians averaged across vectors along axis."""
    A = np.asarray(a)
    B = np.asarray(b)
    num = np.sum(A * B, axis=axis)
    denom = np.linalg.norm(A, axis=axis) * np.linalg.norm(B, axis=axis)
    ang = np.arccos(np.clip(num / (denom + 1e-12), -1.0, 1.0))
    return float(np.mean(ang))


