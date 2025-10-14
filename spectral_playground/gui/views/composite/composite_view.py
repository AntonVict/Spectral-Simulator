"""REFACTORED: Main composite view for RGB image display with spectral analysis tools."""

from __future__ import annotations
from typing import Iterable, Optional
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ...state import PlaygroundState
from .enums import SpectralMode
from .visual_settings import VisualSettingsManager
from .spectral_analysis import SpectralAnalysisTools
from .rendering import RGBRenderer, CacheManager
from .interaction import ZoomManager, PixelInfoDisplay
from .overlays import ObjectOverlayManager
from .toolbar import CompositeToolbar


class CompositeView:
    """Matplotlib-backed view for the composite image display (REFACTORED)."""

    def __init__(self, parent: tk.Widget, on_visual_settings_changed: Optional[callable] = None) -> None:
        """Initialize the composite view.
        
        Args:
            parent: Parent widget
            on_visual_settings_changed: Callback when visual settings change
        """
        # Create figure and canvas
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.figure.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.06)
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=1, column=0, sticky=tk.NSEW)
        
        # Configure parent grid weights
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        # Initialize state
        self._current_data: Optional[PlaygroundState] = None
        self._current_channels: Optional[tuple] = None
        self._image_artist = None  # Store the imshow artist for updates
        self._is_updating: bool = False  # Flag to prevent zoom saves during redraws
        self.spectral_mode = SpectralMode.NONE
        
        # Initialize component managers
        self.visual_settings = VisualSettingsManager(
            self.canvas_widget, 
            on_visual_settings_changed or self._redraw_composite
        )
        self.spectral_tools = SpectralAnalysisTools(lambda: self._current_data)
        self.renderer = RGBRenderer(self.visual_settings)
        self.cache = CacheManager(max_cache_size=20)
        self.zoom_manager = ZoomManager()
        self.object_overlay = ObjectOverlayManager()
        
        # Create toolbar (after managers are initialized)
        self.toolbar_manager = CompositeToolbar(
            parent=parent,
            figure=self.figure,
            canvas=self.canvas,
            spectral_mode_callback=self._set_spectral_mode,
            visual_settings_callback=self.visual_settings.show_settings_dialog,
            object_overlay_callback=self._toggle_object_overlay,
            area_select_callback=self._toggle_area_select_mode,
            show_objects_var=self.object_overlay.show_objects
        )
        
        # Override home button to use zoom manager
        self.toolbar_manager.override_home_button(self._custom_home)
        
        # Create pixel info display
        self.pixel_info = PixelInfoDisplay(self.toolbar_manager.toolbar_frame)
        
        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_motion)
        self.canvas.mpl_connect('draw_event', self._on_draw)

    @property
    def latest_rgb(self) -> Optional[np.ndarray]:
        """Get the latest rendered RGB image."""
        return self.cache.current_rgb
    
    @property
    def show_objects(self):
        """Compatibility property for object overlay show state."""
        return self.object_overlay.show_objects
    
    def set_object_selection_callback(self, callback: callable) -> None:
        """Set callback for when objects are selected."""
        self.object_overlay.set_selection_callback(callback)
    
    def select_objects(self, object_ids: list) -> None:
        """Set selected objects and update display."""
        self.object_overlay.selected_object_ids = object_ids
        
        if self.object_overlay.show_objects.get():
            self._update_object_overlay_fast()
            self.canvas.draw_idle()
        
        if self.object_overlay.on_selection_changed:
            self.object_overlay.on_selection_changed(object_ids)
    
    def get_selected_objects(self) -> list:
        """Get currently selected object IDs."""
        return self.object_overlay.get_selected_objects()

    def update(self, state: PlaygroundState, active_channels: Iterable[bool]) -> None:
        """Update the composite image display (OPTIMIZED).
        
        Args:
            state: Current playground state
            active_channels: Which channels are active for display
        """
        self._is_updating = True
        self._current_data = state
        data = state.data
        
        if not data.has_data:
            self.figure.clear()
            self.figure.suptitle('No dataset loaded', fontsize=12)
            self.zoom_manager.clear_state()
            self.cache.clear()
            self._image_artist = None
            self._clear_overlays()
            self.canvas.draw_idle()
            self._is_updating = False
            return

        # Convert to tuple for caching
        channel_tuple = tuple(active_channels)
        
        # Check if dataset instance changed
        dataset_changed = self.cache.check_dataset_changed(data)
        
        # Check if image dimensions changed
        current_shape = data.field.shape
        image_dimensions_changed = self.zoom_manager.image_dimensions_changed(current_shape)
        
        # Check if channels changed
        channels_changed = (self._current_channels != channel_tuple)
        
        # Check if cache was cleared (e.g., by visual settings change)
        cache_was_cleared = (not self.cache.is_cached(channel_tuple)) and (self._current_channels is not None)
        
        # Save current zoom if image dimensions haven't changed
        if not image_dimensions_changed and not self._is_updating:
            axes = self.figure.get_axes()
            if axes:
                self.zoom_manager.save_current_limits(axes[0])
        
        if image_dimensions_changed:
            # New image size - full reset required
            H, W = current_shape
            self.zoom_manager.set_original_extent(H, W)
            self.zoom_manager.reset_zoom()
            self.cache.clear()
            
            # Full redraw needed
            self._full_redraw(data, channel_tuple)
        elif channels_changed or cache_was_cleared or dataset_changed:
            # Channels changed OR cache cleared OR dataset changed - smart RGB update
            self._update_rgb_only(data, channel_tuple)
        else:
            # Nothing changed
            pass
        
        self._current_channels = channel_tuple
        self._is_updating = False

    def _full_redraw(self, data, channel_tuple: tuple) -> None:
        """Perform a full redraw (new dataset or first draw)."""
        # Clear and redraw
        self.figure.clear()
        self._image_artist = None
        self._clear_overlays()

        # Render composite image (with caching)
        rgb = self._get_or_render_rgb(data, channel_tuple)

        # Display image
        ax = self.figure.add_subplot(1, 1, 1)
        self._image_artist = ax.imshow(rgb, aspect='equal')
        ax.set_title('Composite Image', fontsize=14)
        ax.axis('off')
        ax.format_coord = lambda x, y: ''  # Disable matplotlib's coordinate display
        
        # Restore zoom/pan state
        self.zoom_manager.restore_limits(ax)
        
        # Update toolbar's navigation stack
        try:
            self.toolbar_manager.toolbar.push_current()
        except:
            pass
        
        # Sync object overlay state
        self._sync_object_overlay(force_redraw=True)

        self.canvas.draw_idle()

    def _update_rgb_only(self, data, channel_tuple: tuple) -> None:
        """Update RGB without full redraw (channels changed only)."""
        # Get or render RGB
        rgb = self._get_or_render_rgb(data, channel_tuple)
        
        # Update image data without clearing figure
        axes = self.figure.get_axes()
        if axes and self._image_artist:
            self._image_artist.set_data(rgb)
        else:
            # Fallback to full redraw if no axes/artist
            self._full_redraw(data, channel_tuple)
            return
        
        # Sync object overlay state (no force redraw for RGB-only updates)
        self._sync_object_overlay()
        
        self.canvas.draw_idle()

    def _get_or_render_rgb(self, data, channel_tuple: tuple) -> np.ndarray:
        """Get RGB from cache or render if not cached."""
        cached = self.cache.get_cached(channel_tuple)
        if cached is not None:
            return cached
        
        # Render new RGB
        rgb = self.renderer.render_composite(data, list(channel_tuple))
        
        # Cache it
        self.cache.store(channel_tuple, rgb)
        return rgb

    def _redraw_composite(self) -> None:
        """Redraw the composite image with current settings (fallback method)."""
        if not self._current_data or not self._current_data.data.has_data:
            return
        
        # Try to get active channels from parent viewer
        try:
            parent_viewer = self.canvas_widget.master
            while parent_viewer and not hasattr(parent_viewer, 'active_channel_flags'):
                parent_viewer = parent_viewer.master
            
            if parent_viewer and hasattr(parent_viewer, 'active_channel_flags'):
                active_channels = parent_viewer.active_channel_flags()
            else:
                active_channels = [True] * self._current_data.data.Y.shape[0]
            
            # Clear RGB cache to force re-render with new settings
            self.cache.clear()
            self.update(self._current_data, active_channels)
        except:
            if self._current_data and self._current_data.data.has_data:
                active_channels = [True] * self._current_data.data.Y.shape[0]
                self.cache.clear()
                self.update(self._current_data, active_channels)

    def show_expanded(self, parent: tk.Tk) -> None:
        """Show expanded view of the composite image."""
        if self.cache.current_rgb is None:
            return
        
        window = tk.Toplevel(parent)
        window.title('Composite Image (Expanded)')
        window.geometry('1200x900')

        main_frame = tk.Frame(window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        figure = Figure(figsize=(12, 8), dpi=100)
        figure.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05)
        canvas = FigureCanvasTkAgg(figure, master=main_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar_frame = tk.Frame(main_frame)
        toolbar_frame.pack(fill=tk.X, pady=(5, 0))
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

        ax = figure.add_subplot(1, 1, 1)
        ax.imshow(self.cache.current_rgb, aspect='equal')
        ax.set_title('Composite Image (Expanded View)', fontsize=16)
        ax.axis('off')
        ax.format_coord = lambda x, y: ''

        canvas.draw()
    
    # ------------------------------------------------------------------
    # Spectral Mode & Toolbar Callbacks
    # ------------------------------------------------------------------
    
    def _set_spectral_mode(self, mode: SpectralMode) -> None:
        """Set the current spectral analysis mode and update button states."""
        self.spectral_mode = mode
        self.spectral_tools.reset_state()
        self._clear_overlays()
        
        # Deactivate matplotlib toolbar tools
        self.toolbar_manager.deactivate_tools()
        
        # Update button appearances
        self.toolbar_manager.update_spectral_mode_buttons(mode)
    
    def _custom_home(self, *args, **kwargs) -> None:
        """Custom home function that resets to true original view."""
        original_xlim, original_ylim = self.zoom_manager.get_original_limits()
        if original_xlim is not None and original_ylim is not None:
            axes = self.figure.get_axes()
            if axes:
                ax = axes[0]
                ax.set_xlim(original_xlim)
                ax.set_ylim(original_ylim)
                self.canvas.draw_idle()
                # Clear the toolbar's navigation stack
                try:
                    self.toolbar_manager.toolbar._views.clear()
                    self.toolbar_manager.toolbar._positions.clear()
                    self.toolbar_manager.toolbar.push_current()
                except:
                    pass
    
    # ------------------------------------------------------------------
    # Mouse Event Handling
    # ------------------------------------------------------------------
    
    def _on_mouse_press(self, event) -> None:
        """Handle mouse press events for spectral analysis and object selection."""
        if event.button != 1:
            return
        if not event.inaxes or not self._current_data or not self._current_data.data.has_data:
            return
        
        # Check if we're in area selection mode for objects
        if self.object_overlay.area_select_mode:
            self.object_overlay.start_area_selection(event)
            return
        
        # Check if we should handle object selection
        if self.spectral_mode == SpectralMode.NONE:
            # Only handle object clicks if overlay is enabled
            if not self.object_overlay.show_objects.get():
                return
            
            axes = self.figure.get_axes()
            if not axes:
                return
            ax = axes[0]
            
            nearest_id = self.object_overlay.find_nearest_object(
                event.xdata, event.ydata, self._current_data, ax, max_distance=10.0
            )
            
            if nearest_id is not None:
                # Handle multi-select with Ctrl/Cmd
                if event.key in ('control', 'cmd'):
                    # Toggle selection
                    if nearest_id in self.object_overlay.selected_object_ids:
                        self.object_overlay.selected_object_ids.remove(nearest_id)
                    else:
                        self.object_overlay.selected_object_ids.append(nearest_id)
                else:
                    # Single select (replace)
                    self.object_overlay.selected_object_ids = [nearest_id]
                
                self.select_objects(self.object_overlay.selected_object_ids)
            else:
                # Clicked on empty space - clear selection
                if not event.key in ('control', 'cmd'):
                    self.select_objects([])
            return
        
        # Handle spectral analysis modes
        if self.spectral_mode == SpectralMode.PIXEL:
            self.spectral_tools.handle_pixel_click(event)
        elif self.spectral_mode == SpectralMode.LINE:
            self._clear_overlays()
            self.spectral_tools.start_line(event)
        elif self.spectral_mode == SpectralMode.AREA:
            self._clear_overlays()
            self.spectral_tools.start_area(event)
    
    def _on_mouse_release(self, event) -> None:
        """Handle mouse release events for spectral analysis and area selection."""
        if event.button != 1:
            return
        
        # Handle area selection for objects
        if self.object_overlay.area_select_mode and self.object_overlay.area_select_start:
            selected_ids = self.object_overlay.complete_area_selection(event, self._current_data)
            
            # Handle multi-select with Ctrl
            if event.key in ('control', 'cmd'):
                # Add to existing selection
                for obj_id in selected_ids:
                    if obj_id not in self.object_overlay.selected_object_ids:
                        self.object_overlay.selected_object_ids.append(obj_id)
            else:
                # Replace selection
                self.object_overlay.selected_object_ids = selected_ids
            
            self.select_objects(self.object_overlay.selected_object_ids)
            return
        
        if not event.inaxes or not self._current_data or not self._current_data.data.has_data:
            return
        
        if self.spectral_mode == SpectralMode.LINE and self.spectral_tools.has_active_line():
            self.spectral_tools.complete_line(event)
            self._clear_overlays()
        elif self.spectral_mode == SpectralMode.AREA and self.spectral_tools.has_active_area():
            self.spectral_tools.complete_area(event)
            self._clear_overlays()
    
    def _on_mouse_motion(self, event) -> None:
        """Handle mouse motion for dynamic overlays and pixel value display."""
        # Handle area selection rectangle update
        if self.object_overlay.area_select_mode and self.object_overlay.area_select_start:
            axes = self.figure.get_axes()
            if axes:
                self.object_overlay.update_area_selection(event, axes[0], self.canvas)
            return
        
        # Update pixel value display
        self.pixel_info.update_from_event(event, self._current_data)
        
        if not event.inaxes or not self._current_data or not self._current_data.data.has_data:
            return
        
        # Handle spectral mode overlays
        if self.spectral_mode == SpectralMode.LINE and self.spectral_tools.has_active_line():
            self._clear_overlays()
            ax = self.figure.axes[0]
            overlay = self.spectral_tools.update_line_preview(event, ax, self.canvas)
            if overlay:
                self.spectral_tools.set_current_overlay(overlay)
                self.canvas.draw_idle()
        elif self.spectral_mode == SpectralMode.AREA and self.spectral_tools.has_active_area():
            self._clear_overlays()
            ax = self.figure.axes[0]
            overlay = self.spectral_tools.update_area_preview(event, ax, self.canvas)
            if overlay:
                self.spectral_tools.set_current_overlay(overlay)
                self.canvas.draw_idle()

    def _on_draw(self, event) -> None:
        """Handle draw events to save zoom/pan state."""
        if self._is_updating:
            return
        
        # Only save zoom state, don't trigger overlay redraws
        # (overlay updates are handled explicitly by update methods)
        axes = self.figure.get_axes()
        if axes:
            ax = axes[0]
            self.zoom_manager.save_current_limits(ax)
    
    def _clear_overlays(self) -> None:
        """Clear any drawn overlays on the image without affecting zoom."""
        overlay = self.spectral_tools.get_current_overlay()
        if overlay:
            try:
                overlay.remove()
            except ValueError:
                pass
            self.spectral_tools.clear_overlay()
            self.canvas.draw_idle()
    
    # ------------------------------------------------------------------
    # Object Overlay Management
    # ------------------------------------------------------------------
    
    def _sync_object_overlay(self, force_redraw: bool = False) -> None:
        """Synchronize object overlay state with UI (SINGLE SOURCE OF TRUTH).
        
        This is the ONLY method that should manage overlay visibility.
        All other methods should call this instead of directly manipulating the overlay.
        
        Args:
            force_redraw: If True, force a full redraw even if already showing
        """
        should_show = self.object_overlay.show_objects.get()
        is_showing = (self.object_overlay.object_collection is not None)
        
        axes = self.figure.get_axes()
        if not axes:
            return
        
        ax = axes[0]
        
        if should_show and (not is_showing or force_redraw):
            # Need to show or refresh: draw it
            self.object_overlay.draw_objects(ax, self._current_data, self.zoom_manager)
        elif not should_show and is_showing:
            # Need to hide: clear it
            self.object_overlay.clear_overlay()
        # else: state is already correct, do nothing
    
    def _toggle_object_overlay(self) -> None:
        """Toggle object position overlay."""
        # Just sync state and redraw canvas
        self._sync_object_overlay()
        self.canvas.draw_idle()
    
    def _draw_object_overlay_fast(self) -> None:
        """Draw object positions using fast PathCollection (legacy method)."""
        # Redirect to sync method
        self._sync_object_overlay(force_redraw=True)
    
    def _update_object_overlay_fast(self) -> None:
        """Update object overlay without full redraw (legacy method)."""
        # Redirect to sync method
        self._sync_object_overlay()
    
    def enable_object_overlay(self) -> None:
        """Enable object overlay display (public method for external control).
        
        This can be called by other components (e.g., statistics view, inspector)
        to ensure the overlay is visible when showing specific objects.
        """
        if not self.object_overlay.show_objects.get():
            self.object_overlay.show_objects.set(True)
            self._sync_object_overlay()
            self.canvas.draw_idle()
    
    def _toggle_area_select_mode(self) -> None:
        """Toggle area selection mode for objects."""
        self.object_overlay.toggle_area_select_mode()
        
        if self.object_overlay.area_select_mode:
            # Auto-enable object overlay
            self.enable_object_overlay()
            # Deactivate spectral modes
            self._set_spectral_mode(SpectralMode.NONE)
        
        # Update button appearance
        self.toolbar_manager.update_area_select_button(self.object_overlay.area_select_mode)
