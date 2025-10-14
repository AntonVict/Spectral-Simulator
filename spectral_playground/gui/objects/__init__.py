"""Object management module for spectral playground GUI."""

from .manager import ObjectLayersManager
from .templates import TemplateManager, ObjectTemplate, FluorophoreComponent
from .presets import PresetGenerator
from .dialogs import TemplateEditorDialog
from .ui_components import ObjectLayersUI

__all__ = [
    'ObjectLayersManager',
    'TemplateManager',
    'ObjectTemplate',
    'FluorophoreComponent',
    'PresetGenerator',
    'TemplateEditorDialog',
    'ObjectLayersUI',
]

