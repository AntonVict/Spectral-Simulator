from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np
from numpy.linalg import lstsq
from sklearn.linear_model import Lasso


Array = np.ndarray


class NNLSUnmixer:
    name = "nnls"
    supports_blind = False

    def __init__(self) -> None:
        self._M: Optional[Array] = None

    def fit(self, Y: Array, *, M: Optional[Array] = None, priors: Optional[dict] = None, **kwargs) -> None:
        if M is None:
            raise ValueError("NNLS requires known M")
        self._M = M.astype(np.float32)

    def transform(self, Y: Array, *, M: Optional[Array] = None) -> Dict[str, Any]:
        M_use = self._M if self._M is not None else M
        if M_use is None:
            raise ValueError("NNLS requires known M")
        L, P = Y.shape
        Lm, K = M_use.shape
        if Lm != L:
            raise ValueError("Y and M have incompatible shapes")
        A = np.zeros((K, P), dtype=np.float32)
        # Solve per pixel with nonnegativity using projected gradient (simple)
        Mt = M_use.T
        MtM = Mt @ M_use
        MtY = Mt @ Y
        # Iterative projected gradient descent
        step = 1.0 / (np.linalg.norm(MtM, 2) + 1e-6)
        iters = 200
        for p in range(P):
            a = np.zeros(K, dtype=np.float32)
            b = MtY[:, p]
            for _ in range(iters):
                grad = MtM @ a - b
                a = a - step * grad
                a = np.maximum(a, 0.0)
            A[:, p] = a
        return {"A": A}


class LASSOUnmixer:
    name = "lasso"
    supports_blind = False

    def __init__(self, alpha: float = 0.01, positive: bool = True) -> None:
        self.alpha = float(alpha)
        self.positive = bool(positive)
        self._M: Optional[Array] = None

    def fit(self, Y: Array, *, M: Optional[Array] = None, priors: Optional[dict] = None, **kwargs) -> None:
        if M is None:
            raise ValueError("LASSO requires known M")
        self._M = M.astype(np.float32)

    def transform(self, Y: Array, *, M: Optional[Array] = None) -> Dict[str, Any]:
        M_use = self._M if self._M is not None else M
        if M_use is None:
            raise ValueError("LASSO requires known M")
        L, P = Y.shape
        Lm, K = M_use.shape
        if Lm != L:
            raise ValueError("Y and M have incompatible shapes")
        A = np.zeros((K, P), dtype=np.float32)
        clf = Lasso(alpha=self.alpha, fit_intercept=False, positive=self.positive, max_iter=5000)
        for p in range(P):
            clf.fit(M_use, Y[:, p])
            A[:, p] = clf.coef_.astype(np.float32)
        return {"A": A}


