from __future__ import annotations

from typing import Tuple


def wavelength_to_rgb_nm(nm: float) -> Tuple[float, float, float]:
    """Approximate mapping from wavelength in nm to RGB tuple."""
    w = nm
    if w < 380 or w > 780:
        return (0.5, 0.5, 0.5)
    if w < 440:
        r = -(w - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif w < 490:
        r = 0.0
        g = (w - 440) / (490 - 440)
        b = 1.0
    elif w < 510:
        r = 0.0
        g = 1.0
        b = -(w - 510) / (510 - 490)
    elif w < 580:
        r = (w - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif w < 645:
        r = 1.0
        g = -(w - 645) / (645 - 580)
        b = 0.0
    else:
        r = 1.0
        g = 0.0
        b = 0.0

    if w < 420:
        f = 0.3 + 0.7 * (w - 380) / (420 - 380)
    elif w > 700:
        f = 0.3 + 0.7 * (780 - w) / (780 - 700)
    else:
        f = 1.0
    return (float(r * f), float(g * f), float(b * f))
