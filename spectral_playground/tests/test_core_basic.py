from __future__ import annotations

import numpy as np

from spectral_playground.core.spectra import SpectralSystem, Channel, Fluorophore


def test_spectral_pdfs_normalize():
    lambdas = np.arange(450, 701, 1, dtype=np.float32)
    channels = [Channel(name="C1", center_nm=550, bandwidth_nm=50)]
    fluors = [
        Fluorophore(name="F1", model="gaussian", params={"mu": 560, "sigma": 10}, brightness=1.0),
        Fluorophore(name="F2", model="lognormal", params={"mu": 6.2, "sigma": 0.08}, brightness=1.0),
        Fluorophore(name="F3", model="skewnorm", params={"mu": 520, "sigma": 12, "alpha": 4}, brightness=1.0),
    ]
    spec = SpectralSystem(lambdas=lambdas, channels=channels, fluors=fluors)
    # Each spectral PDF integrates to 1 across the grid
    step = spec.lambdas[1] - spec.lambdas[0]
    for f in fluors:
        pdf = spec._pdf(f)
        area = float(np.sum(pdf) * step)
        assert np.isclose(area, 1.0, atol=1e-3)


