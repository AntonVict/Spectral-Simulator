"""Composite image view module.

This module provides the composite image visualization with:
- Real-time RGB composite rendering
- Visual settings (normalization, gamma, percentile, log scaling)
- Spectral analysis tools (pixel, line, area profiling)
"""

from .composite_view import CompositeView
from .enums import SpectralMode

__all__ = ['CompositeView', 'SpectralMode']
