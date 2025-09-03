from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np


def save_npz(path: str | Path, arrays: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **arrays)


def load_npz(path: str | Path) -> dict:
    with np.load(Path(path), allow_pickle=True) as data:
        out = {k: data[k] for k in data.files}
    return out


