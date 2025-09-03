from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np


Array = np.ndarray


@dataclass
class SynthDataset:
    Y: Array  # (L, P)
    M: Array  # (L, K)
    A: Array  # (K, P)
    B: Array  # (L, P)
    S: Optional[Array]  # (L, L) optional, unused in MVP
    meta: Dict[str, Any]


