"""Object management module for spectral playground GUI."""

from .manager import ObjectLayersManager
from .presets import PresetGenerator
from .ui_components import ObjectLayersUI
from .composition import CompositionGenerator

__all__ = [
    'ObjectLayersManager',
    'PresetGenerator',
    'ObjectLayersUI',
    'CompositionGenerator',
]

