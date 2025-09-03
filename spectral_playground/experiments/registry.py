from __future__ import annotations

from typing import Dict, Any

from ..unmix.linear import NNLSUnmixer, LASSOUnmixer
from ..unmix.nmf import NMFUnmixer
from ..unmix.em import PoissonEMUnmixer


def make_unmixer(spec: Dict[str, Any]):
    method = spec.get("method")
    if method == "nnls":
        return NNLSUnmixer()
    if method == "lasso":
        alpha = float(spec.get("alpha", 0.01))
        return LASSOUnmixer(alpha=alpha)
    if method == "nmf":
        n_iter = int(spec.get("n_iter", 200))
        beta_div = float(spec.get("beta_div", 2.0))
        K = spec.get("K")
        return NMFUnmixer(n_iter=n_iter, beta_div=beta_div, K=K)
    if method == "em_poisson":
        n_iter = int(spec.get("n_iter", 200))
        return PoissonEMUnmixer(n_iter=n_iter)
    raise ValueError(f"Unknown method: {method}")


