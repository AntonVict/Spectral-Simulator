from __future__ import annotations

import numpy as np

from spectral_playground.core.spectra import SpectralSystem, Channel, Fluorophore
from spectral_playground.unmix.linear import NNLSUnmixer, LASSOUnmixer


def _toy():
    lambdas = np.arange(450, 701, 1, dtype=np.float32)
    channels = [
        Channel(name="C1", center_nm=500, bandwidth_nm=30),
        Channel(name="C2", center_nm=550, bandwidth_nm=30),
        Channel(name="C3", center_nm=600, bandwidth_nm=30),
    ]
    fluors = [
        Fluorophore(name="F1", model="gaussian", params={"mu": 520, "sigma": 12}, brightness=1.0),
        Fluorophore(name="F2", model="skewnorm", params={"mu": 560, "sigma": 10, "alpha": 6}, brightness=0.8),
    ]
    spec = SpectralSystem(lambdas=lambdas, channels=channels, fluors=fluors)
    M = spec.build_M()
    return M


def test_nnls_basic():
    M = _toy()
    K = M.shape[1]
    P = 100
    rng = np.random.default_rng(0)
    A = rng.random((K, P), dtype=float).astype(np.float32)
    Y = M @ A
    un = NNLSUnmixer()
    un.fit(Y, M=M)
    out = un.transform(Y, M=M)
    A_hat = out["A"]
    rel = np.linalg.norm(A_hat - A) / (np.linalg.norm(A) + 1e-9)
    assert rel < 0.2


def test_lasso_positive():
    M = _toy()
    K = M.shape[1]
    P = 50
    rng = np.random.default_rng(1)
    A = rng.random((K, P), dtype=float).astype(np.float32)
    Y = M @ A
    un = LASSOUnmixer(alpha=0.001)
    un.fit(Y, M=M)
    out = un.transform(Y, M=M)
    A_hat = out["A"]
    assert np.all(A_hat >= -1e-6)


