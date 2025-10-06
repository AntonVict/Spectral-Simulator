"""Object position overlay and selection management."""

from __future__ import annotations
from typing import Optional, List, TYPE_CHECKING
import numpy as np
import tkinter as tk
from matplotlib.collections import PathCollection
from matplotlib.patches import Rectangle

if TYPE_CHECKING:
    from ....state import PlaygroundState


class ObjectOverlayManager:
    """Manages object position overlay and selection."""
    
    def __init__(self):
        """Initialize object overlay manager."""
        self.show_objects = tk.BooleanVar(value=False)
        self.object_collection: Optional[PathCollection] = None
        self.selected_object_ids: List[int] = []
        self.on_selection_changed: Optional[callable] = None
        
        # Area selection state
        self.area_select_mode = False
        self.area_select_start: Optional[tuple] = None
        self.area_select_rect: Optional[Rectangle] = None
    
    def set_selection_callback(self, callback: callable) -> None:
        """Set callback for when objects are selected.
        
        Args:
            callback: Callback function(object_ids)
        """
        self.on_selection_changed = callback
    
    def toggle_display(self, canvas) -> None:
        """Toggle object overlay display.
        
        Args:
            canvas: Matplotlib canvas
        """
        if not self.show_objects.get():
            self.clear_overlay()
        canvas.draw_idle()
    
    def draw_objects(self, ax, state: 'PlaygroundState', zoom_manager) -> None:
        """Draw object positions (optimized with frustum culling).
        
        Args:
            ax: Matplotlib axes
            state: Current playground state
            zoom_manager: Zoom manager for viewport culling
        """
        self.clear_overlay()
        
        if not state or not state.data.has_data:
            return
        
        visible_objects = self._get_visible_objects(ax, state)
        if not visible_objects:
            return
        
        # Determine marker size based on zoom level
        zoom_level = self._calculate_zoom_level(ax, zoom_manager)
        if zoom_level == 'far':
            markersize, linewidth = 2, 0.5
        elif zoom_level == 'medium':
            markersize, linewidth = 4, 1.0
        else:
            markersize, linewidth = 6, 1.5
        
        # Prepare positions and colors
        positions, colors, sizes = [], [], []
        
        for obj in visible_objects:
            pos = obj.get('position', (0, 0))
            y, x = pos
            positions.append([x, y])
            
            obj_id = obj['id']
            if obj_id in self.selected_object_ids:
                colors.append('yellow')
                sizes.append(markersize * 1.5)
            else:
                colors.append('cyan')
                sizes.append(markersize)
        
        if not positions:
            return
        
        positions = np.array(positions)
        
        # Create single PathCollection for all objects (optimized)
        self.object_collection = ax.scatter(
            positions[:, 0], positions[:, 1],
            c=colors, s=sizes, marker='o',
            facecolors='none', edgecolors=colors,
            linewidths=linewidth, picker=True
        )
    
    def update_objects(self, ax, state: 'PlaygroundState', zoom_manager) -> None:
        """Update object overlay without full redraw.
        
        Args:
            ax: Matplotlib axes
            state: Current playground state
            zoom_manager: Zoom manager
        """
        if not self.show_objects.get():
            return
        
        # Remove old collection
        if self.object_collection:
            try:
                self.object_collection.remove()
            except ValueError:
                pass
            self.object_collection = None
        
        # Redraw with visible objects only
        self.draw_objects(ax, state, zoom_manager)
    
    def clear_overlay(self) -> None:
        """Remove object markers from display."""
        if self.object_collection:
            try:
                self.object_collection.remove()
            except ValueError:
                pass
            self.object_collection = None
    
    def find_nearest_object(self, x: float, y: float, state: 'PlaygroundState', 
                           ax, max_distance: float = 10.0) -> Optional[int]:
        """Find the nearest object to clicked position.
        
        Args:
            x, y: Click position in data coordinates
            state: Current playground state
            ax: Matplotlib axes
            max_distance: Maximum distance in pixels to consider
            
        Returns:
            Object ID if found within max_distance, None otherwise
        """
        if not state or not state.data.has_data:
            return None
        
        visible_objects = self._get_visible_objects(ax, state)
        if not visible_objects:
            return None
        
        min_dist = float('inf')
        nearest_id = None
        
        for obj in visible_objects:
            pos = obj.get('position', (0, 0))
            obj_y, obj_x = pos
            
            dist = np.sqrt((x - obj_x)**2 + (y - obj_y)**2)
            
            if dist < min_dist and dist <= max_distance:
                min_dist = dist
                nearest_id = obj['id']
        
        return nearest_id
    
    def select_objects(self, object_ids: List[int], canvas) -> None:
        """Set selected objects and notify callback.
        
        Args:
            object_ids: List of object IDs to select
            canvas: Matplotlib canvas for redraw
        """
        self.selected_object_ids = object_ids
        canvas.draw_idle()
        
        if self.on_selection_changed:
            self.on_selection_changed(object_ids)
    
    def get_selected_objects(self) -> List[int]:
        """Get currently selected object IDs.
        
        Returns:
            List of selected object IDs
        """
        return self.selected_object_ids.copy()
    
    def toggle_area_select_mode(self) -> None:
        """Toggle area selection mode."""
        self.area_select_mode = not self.area_select_mode
        if not self.area_select_mode:
            self._clear_area_selection_rect()
    
    def start_area_selection(self, event) -> None:
        """Start area selection rectangle.
        
        Args:
            event: Matplotlib mouse event
        """
        if not event.inaxes:
            return
        self.area_select_start = (event.xdata, event.ydata)
        self._clear_area_selection_rect()
    
    def update_area_selection(self, event, ax, canvas) -> None:
        """Update area selection rectangle during drag.
        
        Args:
            event: Matplotlib mouse event
            ax: Matplotlib axes
            canvas: Matplotlib canvas
        """
        if not self.area_select_start or not event.inaxes:
            return
        
        self._clear_area_selection_rect()
        
        x0, y0 = self.area_select_start
        x1, y1 = event.xdata, event.ydata
        
        width = x1 - x0
        height = y1 - y0
        
        self.area_select_rect = Rectangle((x0, y0), width, height,
                                         linewidth=2, edgecolor='yellow',
                                         facecolor='yellow', alpha=0.2)
        ax.add_patch(self.area_select_rect)
        canvas.draw_idle()
    
    def complete_area_selection(self, event, state: 'PlaygroundState') -> List[int]:
        """Complete area selection and return selected object IDs.
        
        Args:
            event: Matplotlib mouse event
            state: Current playground state
            
        Returns:
            List of selected object IDs
        """
        if not self.area_select_start or not event.inaxes:
            self._clear_area_selection_rect()
            self.area_select_start = None
            return []
        
        x0, y0 = self.area_select_start
        x1, y1 = event.xdata, event.ydata
        
        xmin, xmax = min(x0, x1), max(x0, x1)
        ymin, ymax = min(y0, y1), max(y0, y1)
        
        selected_ids = []
        if state and state.data.has_data:
            objects = state.data.metadata.get('objects', [])
            for obj in objects:
                y, x = obj['position']
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    selected_ids.append(obj['id'])
        
        self._clear_area_selection_rect()
        self.area_select_start = None
        return selected_ids
    
    def _clear_area_selection_rect(self) -> None:
        """Clear area selection rectangle."""
        if self.area_select_rect:
            try:
                self.area_select_rect.remove()
            except ValueError:
                pass
            self.area_select_rect = None
    
    def _get_visible_objects(self, ax, state: 'PlaygroundState') -> List[dict]:
        """Return only objects within current viewport (frustum culling).
        
        Args:
            ax: Matplotlib axes
            state: Current playground state
            
        Returns:
            List of visible objects
        """
        if not state or not state.data.has_data:
            return []
        
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        all_objects = state.data.metadata.get('objects', [])
        
        margin = 5.0
        visible = []
        for obj in all_objects:
            y, x = obj['position']
            if (xlim[0] - margin) <= x <= (xlim[1] + margin) and \
               (ylim[1] - margin) <= y <= (ylim[0] + margin):
                visible.append(obj)
        
        return visible
    
    def _calculate_zoom_level(self, ax, zoom_manager) -> str:
        """Calculate current zoom level for adaptive rendering.
        
        Args:
            ax: Matplotlib axes
            zoom_manager: Zoom manager
            
        Returns:
            Zoom level: 'far', 'medium', or 'near'
        """
        original_xlim, _ = zoom_manager.get_original_limits()
        if not original_xlim:
            return 'full'
        
        xlim = ax.get_xlim()
        
        original_width = original_xlim[1] - original_xlim[0]
        current_width = xlim[1] - xlim[0]
        zoom_factor = original_width / current_width
        
        if zoom_factor < 1.5:
            return 'far'
        elif zoom_factor < 5:
            return 'medium'
        else:
            return 'near'

