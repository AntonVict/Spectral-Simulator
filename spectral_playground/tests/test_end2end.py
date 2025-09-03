from __future__ import annotations

import numpy as np

from spectral_playground.core.spectra import SpectralSystem, Channel, Fluorophore
from spectral_playground.core.spatial import FieldSpec, AbundanceField
from spectral_playground.core.background import BackgroundModel
from spectral_playground.core.noise import NoiseModel
from spectral_playground.core.simulate import ForwardConfig, ForwardModel
from spectral_playground.unmix.linear import NNLSUnmixer


def test_end_to_end_nnls_recovers_signal():
    rng = np.random.default_rng(0)
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

    field = FieldSpec(shape=(32, 32), pixel_size_nm=100)
    af = AbundanceField(rng)
    A = af.sample(K=M.shape[1], field=field, kind="dots", density_per_100x100_um2=20)

    bg = BackgroundModel(rng)
    noise = NoiseModel(rng)
    fwd = ForwardModel(spectral=spec, field=field, bg=bg, noise=noise, cfg=ForwardConfig())
    Y = fwd.synthesize(A, B=bg.sample(M.shape[0], *field.shape, kind="constant", level=0.0), noise_params={"gain": 1.0, "read_sigma": 0.5})

    un = NNLSUnmixer()
    un.fit(Y, M=M)
    out = un.transform(Y, M=M)
    A_hat = out["A"]
    # Reconstruction error should be small relative to signal
    Y_hat = M @ A_hat
    rel_err = np.linalg.norm(Y - Y_hat) / (np.linalg.norm(Y) + 1e-9)
    assert rel_err < 0.2


