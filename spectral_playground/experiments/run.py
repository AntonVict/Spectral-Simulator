from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml
import typer

from ..core.spectra import SpectralSystem, Channel, Fluorophore
from ..core.spatial import FieldSpec, AbundanceField
from ..core.background import BackgroundModel
from ..core.noise import NoiseModel
from ..core.simulate import ForwardConfig, ForwardModel
from ..data.dataset import SynthDataset
from ..data.io import save_npz
from ..eval.metrics import rmse, sam
from .registry import make_unmixer


app = typer.Typer(add_completion=False)


def _build_spectral(cfg: dict) -> SpectralSystem:
    g = cfg["grid"]["lambdas"]
    lambdas = np.arange(float(g["start"]), float(g["stop"]) + 1e-9, float(g["step"]))
    channels = []
    for ch in cfg["channels"]:
        channels.append(Channel(name=ch["name"], center_nm=float(ch["center_nm"]), bandwidth_nm=float(ch["bandwidth_nm"])) )
    fluors = []
    for f in cfg["fluors"]:
        fluors.append(Fluorophore(name=f["name"], model=f["model"], params=f["params"], brightness=float(f.get("brightness", 1.0))))
    return SpectralSystem(lambdas=lambdas.astype(np.float32), channels=channels, fluors=fluors)


@app.command()
def main(cfg: str = typer.Option(..., help="Path to YAML config"), seed: int = typer.Option(0, help="Random seed"), outdir: str = typer.Option("outputs", help="Output directory")):
    with open(cfg, "r", encoding="utf-8") as f:
        C = yaml.safe_load(f)

    rng = np.random.default_rng(seed)
    spectral = _build_spectral(C)
    M = spectral.build_M()

    # Spatial/abundances
    H = int(C["spatial"]["field"]["H"]) 
    W = int(C["spatial"]["field"]["W"]) 
    px = float(C["spatial"]["field"]["pixel_size_nm"]) 
    field = FieldSpec(shape=(H, W), pixel_size_nm=px)
    af = AbundanceField(rng)
    A = af.sample(K=M.shape[1], field=field, kind=C["spatial"]["abundances"]["kind"], **{k: v for k, v in C["spatial"]["abundances"].items() if k != "kind"})

    # Background & noise
    bg = BackgroundModel(rng)
    noise = NoiseModel(rng)
    fcfg = ForwardConfig(use_S=bool(C["forward"].get("use_S", False)), use_PSF=bool(C["forward"].get("use_PSF", False)), photon_counting=bool(C["forward"].get("photon_counting", True)))
    fwd = ForwardModel(spectral=spectral, field=field, bg=bg, noise=noise, cfg=fcfg)

    noise_params = C["forward"].get("noise", {"gain": 1.0, "read_sigma": 0.0, "dark_rate": 0.0})
    B = bg.sample(M.shape[0], H, W, kind=C.get("background", {"kind": "constant"}).get("kind", "constant"), **C.get("background", {}))
    Y = fwd.synthesize(A, B=B, noise_params=noise_params)

    ds = SynthDataset(Y=Y, M=M, A=A, B=B, S=None, meta={"seed": seed, "cfg_path": cfg})

    # Run unmixers
    results: list[dict] = []
    for spec in C.get("unmix", []):
        un = make_unmixer(spec)
        known_M = M if spec.get("method") != "nmf" or spec.get("K") is None else None
        un.fit(Y, M=known_M, priors=C.get("priors"))
        out = un.transform(Y, M=known_M)
        A_hat = out.get("A")
        M_hat = out.get("M", known_M)
        metrics = {
            "rmse_Y": rmse(Y, (M_hat @ A_hat) + B) if (A_hat is not None and M_hat is not None) else None,
            "rmse_A": rmse(A_hat, A) if A_hat is not None else None,
            "sam_M": sam(M_hat, M, axis=0) if M_hat is not None else None,
        }
        results.append({"method": un.name, "metrics": metrics})

    # Save
    out_dir = Path(outdir)
    save_npz(out_dir / "dataset.npz", {"Y": Y, "M": M, "A": A, "B": B})
    save_npz(out_dir / "results.npz", {f"{r['method']}_metrics": np.array(list(r["metrics"].values()), dtype=float) for r in results})


if __name__ == "__main__":
    app()


