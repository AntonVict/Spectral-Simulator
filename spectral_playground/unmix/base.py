from __future__ import annotations

from typing import Protocol, Optional, Dict, Any

import numpy as np


Array = np.ndarray


class BaseUnmixer(Protocol):
    name: str
    supports_blind: bool  # if M can be unknown

    def fit(self, Y: Array, *, M: Optional[Array] = None, priors: Optional[dict] = None, **kwargs) -> None:
        ...

    def transform(self, Y: Array, *, M: Optional[Array] = None) -> Dict[str, Any]:
        ...


