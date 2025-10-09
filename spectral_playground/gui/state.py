from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - imports for type hints only
    from spectral_playground.core.spectra import SpectralSystem
    from spectral_playground.core.spatial import FieldSpec
    from spectral_playground.core.geometry import GeometricScene
else:
    SpectralSystem = FieldSpec = Any  # type: ignore
    GeometricScene = Any  # type: ignore


Array = np.ndarray


@dataclass
class PlaygroundData:
    """Container for the dataset currently loaded in the GUI."""

    Y: Optional[Array] = None
    A: Optional[Array] = None
    B: Optional[Array] = None
    M: Optional[Array] = None
    spectral: Optional['SpectralSystem'] = None
    field: Optional['FieldSpec'] = None
    geometric_scene: Optional['GeometricScene'] = None
    metadata: Dict[str, Any] = dataclass_field(default_factory=dict)

    def clear(self) -> None:
        self.Y = None
        self.A = None
        self.B = None
        self.M = None
        self.spectral = None
        self.field = None
        self.geometric_scene = None
        self.metadata = {}

    @property
    def has_data(self) -> bool:
        return self.Y is not None and self.A is not None and self.field is not None and self.spectral is not None
    
    @property
    def has_geometric_data(self) -> bool:
        return self.geometric_scene is not None and len(self.geometric_scene) > 0


@dataclass
class SelectionState:
    """Tracks which channels/fluorophores are currently visible."""

    active_channels: Tuple[bool, ...] = tuple()
    active_fluors: Tuple[bool, ...] = tuple()
    show_measured_total: bool = True

    def update_channels(self, flags: Tuple[bool, ...]) -> None:
        self.active_channels = flags

    def update_fluors(self, flags: Tuple[bool, ...]) -> None:
        self.active_fluors = flags
    
    def set_measured_total(self, visible: bool) -> None:
        self.show_measured_total = visible


@dataclass
class PlaygroundState:
    """Aggregate UI state for the visualization playground."""

    data: PlaygroundData = dataclass_field(default_factory=PlaygroundData)
    selections: SelectionState = dataclass_field(default_factory=SelectionState)
    save_directory: Optional[str] = None

    def reset(self) -> None:
        self.data.clear()
        self.selections = SelectionState()
