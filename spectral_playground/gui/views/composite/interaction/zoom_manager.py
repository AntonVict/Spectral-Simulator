"""Zoom and pan state management."""

from __future__ import annotations
from typing import Optional, Tuple


class ZoomManager:
    """Manages zoom/pan state for the composite view."""
    
    def __init__(self):
        """Initialize zoom manager."""
        self._original_xlim: Optional[Tuple[float, float]] = None  # True original view extent
        self._original_ylim: Optional[Tuple[float, float]] = None
        self._saved_xlim: Optional[Tuple[float, float]] = None  # Current/saved zoom state
        self._saved_ylim: Optional[Tuple[float, float]] = None
        self._last_image_shape: Optional[Tuple[int, int]] = None  # Track image dimensions (H, W)
    
    def set_original_extent(self, H: int, W: int) -> None:
        """Set the original extent for a new image.
        
        Args:
            H: Image height
            W: Image width
        """
        self._original_xlim = (-0.5, W - 0.5)
        self._original_ylim = (H - 0.5, -0.5)  # Inverted for image display
        self._last_image_shape = (H, W)
    
    def reset_zoom(self) -> None:
        """Reset zoom to original extent."""
        self._saved_xlim = None
        self._saved_ylim = None
    
    def clear_state(self) -> None:
        """Clear all zoom state (for new dataset)."""
        self._original_xlim = None
        self._original_ylim = None
        self._saved_xlim = None
        self._saved_ylim = None
        self._last_image_shape = None
    
    def save_current_limits(self, ax) -> None:
        """Save current axis limits from matplotlib axes.
        
        Args:
            ax: Matplotlib axes object
        """
        try:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            # Only save if different from current saved state
            if xlim != self._saved_xlim or ylim != self._saved_ylim:
                self._saved_xlim = xlim
                self._saved_ylim = ylim
        except:
            pass
    
    def restore_limits(self, ax) -> None:
        """Restore saved axis limits to matplotlib axes.
        
        Args:
            ax: Matplotlib axes object
        """
        try:
            if self._saved_xlim is not None and self._saved_ylim is not None:
                ax.set_xlim(self._saved_xlim)
                ax.set_ylim(self._saved_ylim)
            elif self._original_xlim is not None:
                # Fallback to original if no saved state
                ax.set_xlim(self._original_xlim)
                ax.set_ylim(self._original_ylim)
        except:
            pass
    
    def get_original_limits(self) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """Get original extent limits.
        
        Returns:
            Tuple of (xlim, ylim) or (None, None) if not set
        """
        return self._original_xlim, self._original_ylim
    
    def get_saved_limits(self) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """Get saved zoom/pan limits.
        
        Returns:
            Tuple of (xlim, ylim) or (None, None) if not set
        """
        return self._saved_xlim, self._saved_ylim
    
    def has_saved_zoom(self) -> bool:
        """Check if zoom state is saved.
        
        Returns:
            True if zoom state exists
        """
        return self._saved_xlim is not None and self._saved_ylim is not None
    
    def image_dimensions_changed(self, current_shape: Tuple[int, int]) -> bool:
        """Check if image dimensions changed.
        
        Args:
            current_shape: Current image shape (H, W)
            
        Returns:
            True if dimensions changed
        """
        return self._last_image_shape != current_shape

