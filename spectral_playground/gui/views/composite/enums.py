"""Enumerations for composite view."""

from enum import Enum


class SpectralMode(Enum):
    """Different spectral analysis modes."""
    NONE = "none"
    PIXEL = "pixel"
    LINE = "line"
    AREA = "area"
