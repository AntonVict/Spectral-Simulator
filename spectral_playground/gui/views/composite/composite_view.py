"""OPTIMIZED: Main composite view for RGB image display with spectral analysis tools."""

from __future__ import annotations
from typing import Iterable, Optional
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.collections import PathCollection

from ...state import PlaygroundState
from ..utils import wavelength_to_rgb_nm
from .enums import SpectralMode
from .visual_settings import VisualSettingsManager
from .spectral_analysis import SpectralAnalysisTools


def create_composite_rgb(Y: np.ndarray, M: np.ndarray, H: int, W: int,
                         channel_selection: Optional[list] = None,
                         colormap: str = 'turbo',
                         normalize_mode: str = 'global',
                         gamma: float = 1.0,
                         brightness: float = 1.0,
                         contrast: float = 1.0) -> np.ndarray:
    """Create RGB composite from spectral data (standalone function).
    
    Args:
        Y: Spectral data (L, P)
        M: Mixing matrix (L, K)
        H, W: Image dimensions
        channel_selection: Which channels to include (None = all)
        colormap: Color mapping method
        normalize_mode: 'global' or 'per_channel'
        gamma: Gamma correction
        brightness: Brightness adjustment
        contrast: Contrast adjustment
        
    Returns:
        RGB image (H, W, 3)
    """
    L = Y.shape[0]
    
    if channel_selection is None:
        channels = [True] * L
    else:
        channels = channel_selection
    
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    eps = 1e-6
    
    # Simple normalization (can be enhanced based on parameters)
    if normalize_mode == "global":
        global_max = 0.0
        for idx, flag in enumerate(channels):
            if flag:
                channel_image = Y[idx].reshape(H, W)
                global_max = max(global_max, np.max(channel_image))
        scale = global_max + eps
    
    # Process each channel
    for idx, flag in enumerate(channels):
        if not flag:
            continue
        
        channel_image = Y[idx].reshape(H, W)
        
        if normalize_mode == "global":
            normalized = channel_image / scale
        else:  # per_channel
            channel_max = np.max(channel_image) + eps
            normalized = channel_image / channel_max
        
        # Apply gamma
        if gamma != 1.0:
            normalized = np.power(np.clip(normalized, 0.0, 1.0), gamma)
        
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # Simple wavelength-to-RGB (simplified - assumes wavelengths or uses channel index)
        # This is a placeholder - in real implementation would use proper wavelength mapping
        hue = idx / max(1, L - 1)  # Spread across spectrum
        color = np.array([
            np.sin(hue * np.pi) ** 2,
            np.sin((hue + 0.33) * np.pi) ** 2,
            np.sin((hue + 0.67) * np.pi) ** 2
        ], dtype=np.float32)
        
        rgb += normalized[..., None] * color[None, None, :]
    
    # Apply brightness and contrast
    rgb = rgb * brightness
    rgb = np.clip((rgb - 0.5) * contrast + 0.5, 0.0, 1.0)
    
    return rgb


class CompositeView:
    """Matplotlib-backed view for the composite image display with performance optimizations."""

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

        # Initialize state first
        self._rgb_cache_dict: dict = {}  # Cache by channel tuple
        self._rgb_cache: Optional[np.ndarray] = None  # Current RGB
        self._current_data: Optional[PlaygroundState] = None
        self._current_channels: Optional[tuple] = None
        self._image_artist = None  # Store the imshow artist for updates
        self.spectral_mode = SpectralMode.NONE
        
        # Object selection state
        self.show_objects = tk.BooleanVar(value=False)
        self.object_collection: Optional[PathCollection] = None  # Single collection for all objects
        self.selected_object_ids = []
        self.on_object_selection_changed: Optional[callable] = None
        self.object_area_select_mode = False  # Area selection mode for objects
        self.area_select_start = None  # Starting point for area selection
        self.area_select_rect = None  # Rectangle artist for area selection
        
        # Zoom/pan state management (improved)
        self._original_xlim: Optional[tuple] = None  # True original view extent
        self._original_ylim: Optional[tuple] = None
        self._saved_xlim: Optional[tuple] = None  # Current/saved zoom state
        self._saved_ylim: Optional[tuple] = None
        self._last_image_shape: Optional[tuple] = None  # Track image dimensions (H, W)
        self._is_updating: bool = False  # Flag to prevent zoom saves during redraws
        
        # Cache for base composite (before visual settings applied)
        self._base_composite_cache: Optional[np.ndarray] = None
        self._cache_channels: Optional[list] = None
        
        # Initialize managers before creating toolbar (toolbar needs them)
        self.visual_settings = VisualSettingsManager(
            self.canvas_widget, 
            on_visual_settings_changed or self._redraw_composite
        )
        self.spectral_tools = SpectralAnalysisTools(lambda: self._current_data)

        # Create toolbar (uses visual_settings and spectral_tools)
        self._create_toolbar(parent)
        
        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_motion)
        # Connect draw event to track zoom changes
        self.canvas.mpl_connect('draw_event', self._on_draw)

    def _create_toolbar(self, parent: tk.Widget) -> None:
        """Create toolbar with matplotlib tools and spectral tools."""
        toolbar_frame = tk.Frame(parent)
        toolbar_frame.grid(row=2, column=0, sticky=tk.EW, pady=(2, 0))
        
        # Standard matplotlib toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.pack(side=tk.LEFT)
        
        # Override home button to use our stored original extent
        self._override_toolbar_home()
        
        # Disable matplotlib's coordinate display
        try:
            self.toolbar.set_message = lambda s: None
        except:
            pass
        
        # Spectral tools section
        self._create_spectral_tools(toolbar_frame)
        
        # Visual settings button
        self._create_visual_settings_button(toolbar_frame)
        
        # Pixel info display
        self._create_pixel_info_display(toolbar_frame)

    def _create_spectral_tools(self, parent: tk.Frame) -> None:
        """Create spectral analysis tool buttons."""
        separator = ttk.Separator(parent, orient='vertical')
        separator.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 5))
        
        spectral_frame = tk.Frame(parent)
        spectral_frame.pack(side=tk.LEFT)
        
        tk.Label(spectral_frame, text="Spectral:", font=('TkDefaultFont', 8)).pack(side=tk.LEFT, padx=(0, 5))
        
        # Tool buttons
        self.pixel_btn = tk.Button(spectral_frame, text="📍", width=3, relief=tk.RAISED,
                                   command=lambda: self._set_spectral_mode(SpectralMode.PIXEL))
        self.pixel_btn.pack(side=tk.LEFT, padx=1)
        
        self.line_btn = tk.Button(spectral_frame, text="📏", width=3, relief=tk.RAISED,
                                  command=lambda: self._set_spectral_mode(SpectralMode.LINE))
        self.line_btn.pack(side=tk.LEFT, padx=1)
        
        self.area_btn = tk.Button(spectral_frame, text="⬛", width=3, relief=tk.RAISED,
                                  command=lambda: self._set_spectral_mode(SpectralMode.AREA))
        self.area_btn.pack(side=tk.LEFT, padx=1)
        
        self.clear_btn = tk.Button(spectral_frame, text="✕", width=3, relief=tk.RAISED,
                                   command=lambda: self._set_spectral_mode(SpectralMode.NONE))
        self.clear_btn.pack(side=tk.LEFT, padx=(5, 0))
        
        # Object selection tools
        obj_separator = ttk.Separator(parent, orient='vertical')
        obj_separator.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5))
        
        obj_frame = tk.Frame(parent)
        obj_frame.pack(side=tk.LEFT)
        
        tk.Label(obj_frame, text="Objects:", font=('TkDefaultFont', 8)).pack(side=tk.LEFT, padx=(0, 5))
        
        self.show_objects_check = ttk.Checkbutton(obj_frame, text="Show", 
                                                  variable=self.show_objects,
                                                  command=self._toggle_object_overlay)
        self.show_objects_check.pack(side=tk.LEFT, padx=1)
        
        self.area_select_btn = tk.Button(obj_frame, text="📦", width=3, relief=tk.RAISED,
                                         command=self._toggle_area_select_mode)
        self.area_select_btn.pack(side=tk.LEFT, padx=(5, 0))
        tk.Label(obj_frame, text="Area Select", font=('TkDefaultFont', 7)).pack(side=tk.LEFT, padx=(2, 0))

    def _create_visual_settings_button(self, parent: tk.Frame) -> None:
        """Create visual settings button."""
        visual_separator = ttk.Separator(parent, orient='vertical')
        visual_separator.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5))
        
        self.visual_settings_btn = tk.Button(parent, text="⚙️", width=3, relief=tk.RAISED,
                                           command=self.visual_settings.show_settings_dialog)
        self.visual_settings_btn.pack(side=tk.LEFT, padx=(0, 10))

    def _create_pixel_info_display(self, parent: tk.Frame) -> None:
        """Create pixel value display."""
        self.pixel_info_frame = tk.Frame(parent)
        self.pixel_info_frame.pack(side=tk.RIGHT, padx=(10, 0))
        
        tk.Label(self.pixel_info_frame, text="Pixel:", font=('TkDefaultFont', 8)).pack(side=tk.LEFT)
        self.pixel_coords_label = tk.Label(self.pixel_info_frame, text="(-,-)", 
                                          font=('TkDefaultFont', 8), width=8)
        self.pixel_coords_label.pack(side=tk.LEFT, padx=(2, 5))
        
        self.pixel_values_label = tk.Label(self.pixel_info_frame, text="Values: -", 
                                          font=('TkDefaultFont', 8), width=35)
        self.pixel_values_label.pack(side=tk.LEFT)
    
    def _override_toolbar_home(self) -> None:
        """Override toolbar's home button to use our stored original extent."""
        original_home = self.toolbar.home
        
        def custom_home(*args, **kwargs):
            """Custom home function that resets to true original view."""
            if self._original_xlim is not None and self._original_ylim is not None:
                axes = self.figure.get_axes()
                if axes:
                    ax = axes[0]
                    ax.set_xlim(self._original_xlim)
                    ax.set_ylim(self._original_ylim)
                    self.canvas.draw_idle()
                    # Clear the toolbar's navigation stack
                    try:
                        self.toolbar._views.clear()
                        self.toolbar._positions.clear()
                        self.toolbar.push_current()
                    except:
                        pass
            else:
                # Fallback to original behavior if we don't have stored extent
                original_home(*args, **kwargs)
        
        self.toolbar.home = custom_home

    @property
    def latest_rgb(self) -> Optional[np.ndarray]:
        """Get the latest rendered RGB image."""
        return self._rgb_cache

    def update(self, state: PlaygroundState, active_channels: Iterable[bool]) -> None:
        """Update the composite image display (OPTIMIZED).
        
        Args:
            state: Current playground state
            active_channels: Which channels are active for display
        """
        self._is_updating = True  # Prevent zoom saves during update
        self._current_data = state
        data = state.data
        
        if not data.has_data:
            self.figure.clear()
            self.figure.suptitle('No dataset loaded', fontsize=12)
            self._saved_xlim = None
            self._saved_ylim = None
            self._original_xlim = None
            self._original_ylim = None
            self._last_image_shape = None
            self._rgb_cache = None
            self._rgb_cache_dict.clear()
            self._image_artist = None
            self._clear_overlays()
            self.canvas.draw_idle()
            self._is_updating = False
            return

        # Convert to tuple for caching
        channel_tuple = tuple(active_channels)
        
        # Check if image dimensions changed (new dataset)
        current_shape = data.field.shape
        image_dimensions_changed = (self._last_image_shape != current_shape)
        
        # Check if channels changed
        channels_changed = (self._current_channels != channel_tuple)
        
        # Check if cache was cleared (e.g., by visual settings change)
        cache_was_cleared = (channel_tuple not in self._rgb_cache_dict) and (self._current_channels is not None)
        
        # Save current zoom if image dimensions haven't changed
        if not image_dimensions_changed and not self._is_updating:
            self._save_axis_limits()
        
        if image_dimensions_changed:
            # New image size - full reset required
            H, W = current_shape
            self._original_xlim = (-0.5, W - 0.5)
            self._original_ylim = (H - 0.5, -0.5)  # Inverted for image display
            self._saved_xlim = None
            self._saved_ylim = None
            self._last_image_shape = current_shape
            self._rgb_cache_dict.clear()  # Clear cache on new dataset
            
            # Full redraw needed
            self._full_redraw(data, channel_tuple)
        elif channels_changed or cache_was_cleared:
            # Channels changed OR cache was cleared (e.g., visual settings) - smart RGB update
            self._update_rgb_only(data, channel_tuple)
        else:
            # Nothing changed (shouldn't happen, but handle gracefully)
            pass
        
        self._current_channels = channel_tuple
        self._is_updating = False

    def _full_redraw(self, data, channel_tuple: tuple) -> None:
        """Perform a full redraw (new dataset or first draw)."""
        # Clear and redraw
        self.figure.clear()
        self._rgb_cache = None
        self._image_artist = None
        self._clear_overlays()

        # Render composite image (with caching)
        rgb = self._get_or_render_rgb(data, channel_tuple)
        self._rgb_cache = rgb

        # Display image
        ax = self.figure.add_subplot(1, 1, 1)
        self._image_artist = ax.imshow(rgb, aspect='equal')
        ax.set_title('Composite Image', fontsize=14)
        ax.axis('off')
        ax.format_coord = lambda x, y: ''  # Disable matplotlib's coordinate display
        
        # Set original extent if this is the first time
        if self._original_xlim is None:
            H, W = data.field.shape
            self._original_xlim = (-0.5, W - 0.5)
            self._original_ylim = (H - 0.5, -0.5)
        
        # Restore zoom/pan state if we have saved state
        if self._saved_xlim is not None:
            ax.set_xlim(self._saved_xlim)
            ax.set_ylim(self._saved_ylim)
        else:
            # Set to original extent
            ax.set_xlim(self._original_xlim)
            ax.set_ylim(self._original_ylim)
        
        # Update toolbar's navigation stack
        try:
            self.toolbar.push_current()
        except:
            pass
        
        # Redraw object overlay if enabled
        if self.show_objects.get():
            self._draw_object_overlay_fast()

        self.canvas.draw_idle()

    def _update_rgb_only(self, data, channel_tuple: tuple) -> None:
        """Update RGB without full redraw (channels changed only)."""
        # Get or render RGB
        rgb = self._get_or_render_rgb(data, channel_tuple)
        self._rgb_cache = rgb
        
        # Update image data without clearing figure
        axes = self.figure.get_axes()
        if axes and self._image_artist:
            self._image_artist.set_data(rgb)
        else:
            # Fallback to full redraw if no axes/artist
            self._full_redraw(data, channel_tuple)
            return
        
        # Redraw object overlay if enabled (only visible objects)
        if self.show_objects.get():
            self._update_object_overlay_fast()
        
        self.canvas.draw_idle()

    def _get_or_render_rgb(self, data, channel_tuple: tuple) -> np.ndarray:
        """Get RGB from cache or render if not cached."""
        if channel_tuple in self._rgb_cache_dict:
            return self._rgb_cache_dict[channel_tuple]
        
        # Render new RGB
        rgb = self._render_composite(data, list(channel_tuple))
        
        # Cache it (limit cache size to prevent memory issues)
        if len(self._rgb_cache_dict) > 20:
            # Remove oldest entry (simple strategy)
            self._rgb_cache_dict.pop(next(iter(self._rgb_cache_dict)))
        
        self._rgb_cache_dict[channel_tuple] = rgb
        return rgb

    def _render_composite(self, data, channels: list) -> np.ndarray:
        """Render the composite RGB image from spectral data.
        
        Args:
            data: PlaygroundData object
            channels: List of active channel flags
            
        Returns:
            RGB image array (H, W, 3)
        """
        H, W = data.field.shape
        Y = data.Y
        
        if not any(channels):
            channels = [True] * Y.shape[0]

        rgb = np.zeros((H, W, 3), dtype=np.float32)
        eps = 1e-6
        
        # Get intensity mapping settings
        settings = self.visual_settings
        percentile = settings.percentile_threshold.get()
        gamma = settings.gamma_correction.get()
        use_log = settings.use_log_scaling.get()
        
        # Calculate normalization scale based on mode
        if settings.normalization_mode.get() == "global":
            global_max = self._calculate_global_max(Y, channels, percentile, H, W)
            global_scale = global_max + eps
        
        # Process each channel
        for idx, flag in enumerate(channels):
            if not flag:
                continue
            
            channel_image = Y[idx].reshape(H, W)
            
            # Determine normalization scale
            if settings.normalization_mode.get() == "global":
                scale = global_scale
            else:  # per_channel (default)
                if percentile >= 100.0:
                    scale = np.max(channel_image) + eps
                else:
                    scale = np.percentile(channel_image, percentile) + eps
            
            # Apply logarithmic scaling if enabled
            if use_log:
                normalized = np.log1p(channel_image) / np.log1p(scale)
            else:
                normalized = channel_image / scale
            
            # Apply gamma correction
            if gamma != 1.0:
                normalized = np.power(np.clip(normalized, 0.0, 1.0), gamma)
            
            # Final clipping
            normalized = np.clip(normalized, 0.0, 1.0)
            
            # Map to color
            color = np.array(wavelength_to_rgb_nm(data.spectral.channels[idx].center_nm), dtype=np.float32)
            rgb += normalized[..., None] * color[None, None, :]

        return np.clip(rgb, 0.0, 1.0)

    def _calculate_global_max(self, Y: np.ndarray, channels: list, percentile: float, H: int, W: int) -> float:
        """Calculate the global maximum across all active channels."""
        global_max = 0.0
        for idx, flag in enumerate(channels):
            if flag:
                channel_image = Y[idx].reshape(H, W)
                if percentile >= 100.0:
                    channel_max = np.max(channel_image)
                else:
                    channel_max = np.percentile(channel_image, percentile)
                global_max = max(global_max, channel_max)
        return global_max

    def show_expanded(self, parent: tk.Tk) -> None:
        """Show expanded view of the composite image."""
        if self._rgb_cache is None:
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

        toolbar_frame = tk.Frame(main_frame)
        toolbar_frame.pack(fill=tk.X, pady=(5, 0))
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

        ax = figure.add_subplot(1, 1, 1)
        ax.imshow(self._rgb_cache, aspect='equal')
        ax.set_title('Composite Image (Expanded View)', fontsize=16)
        ax.axis('off')
        ax.format_coord = lambda x, y: ''

        canvas.draw()

    def _save_axis_limits(self) -> None:
        """Save current axis limits (zoom/pan state)."""
        if self._is_updating:
            return  # Don't save during updates
        
        try:
            axes = self.figure.get_axes()
            if axes:
                ax = axes[0]
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                
                # Only save if different from current saved state
                if xlim != self._saved_xlim or ylim != self._saved_ylim:
                    self._saved_xlim = xlim
                    self._saved_ylim = ylim
        except:
            pass
    
    def _restore_axis_limits(self, ax) -> None:
        """Restore saved axis limits (zoom/pan state)."""
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
            self._rgb_cache_dict.clear()
            self.update(self._current_data, active_channels)
        except:
            if self._current_data and self._current_data.data.has_data:
                active_channels = [True] * self._current_data.data.Y.shape[0]
                self._rgb_cache_dict.clear()
                self.update(self._current_data, active_channels)

    # ------------------------------------------------------------------
    # Spectral Mode & Mouse Event Handling
    # ------------------------------------------------------------------
    
    def _set_spectral_mode(self, mode: SpectralMode) -> None:
        """Set the current spectral analysis mode and update button states."""
        self.spectral_mode = mode
        self.spectral_tools.reset_state()
        self._clear_overlays()
        
        # Deactivate matplotlib toolbar tools without resetting view
        if hasattr(self.toolbar, '_active') and self.toolbar._active:
            self.toolbar._active = None
        
        # Update button appearances
        buttons = [self.pixel_btn, self.line_btn, self.area_btn, self.clear_btn]
        for btn in buttons:
            btn.config(relief=tk.RAISED, bg='SystemButtonFace')
        
        if mode == SpectralMode.PIXEL:
            self.pixel_btn.config(relief=tk.SUNKEN, bg='lightblue')
        elif mode == SpectralMode.LINE:
            self.line_btn.config(relief=tk.SUNKEN, bg='lightgreen')
        elif mode == SpectralMode.AREA:
            self.area_btn.config(relief=tk.SUNKEN, bg='lightyellow')
    
    def _clear_overlays(self) -> None:
        """Clear any drawn overlays on the image without affecting zoom."""
        overlay = self.spectral_tools.get_current_overlay()
        if overlay:
            try:
                overlay.remove()
            except ValueError:
                pass
            self.spectral_tools.clear_overlay()
            # Use draw_idle() to avoid triggering navigation stack updates
            self.canvas.draw_idle()
    
    def _on_mouse_press(self, event) -> None:
        """Handle mouse press events for spectral analysis and object selection."""
        if event.button != 1:
            return
        if not event.inaxes or not self._current_data or not self._current_data.data.has_data:
            return
        
        # Check if we're in area selection mode for objects
        if self.object_area_select_mode:
            self._start_area_selection(event)
            return
        
        # Check if we should handle object selection
        if self.show_objects.get() and self.spectral_mode == SpectralMode.NONE:
            # Object selection mode
            nearest_id = self._find_nearest_object(event.xdata, event.ydata, max_distance=10.0)
            if nearest_id is not None:
                # Handle multi-select with Ctrl/Cmd
                if event.key in ('control', 'cmd'):
                    # Toggle selection
                    if nearest_id in self.selected_object_ids:
                        self.selected_object_ids.remove(nearest_id)
                    else:
                        self.selected_object_ids.append(nearest_id)
                else:
                    # Single select (replace)
                    self.selected_object_ids = [nearest_id]
                
                self.select_objects(self.selected_object_ids)
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
        if self.object_area_select_mode and self.area_select_start:
            self._complete_area_selection(event)
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
        if self.object_area_select_mode and self.area_select_start:
            self._update_area_selection(event)
            return
        
        if not event.inaxes or not self._current_data or not self._current_data.data.has_data:
            self.pixel_coords_label.config(text="(-,-)")
            self.pixel_values_label.config(text="Values: -")
            return
        
        # Update pixel value display
        self._update_pixel_display(event)
        
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
        """Handle draw events to detect zoom/pan changes and update object overlay."""
        if self._is_updating:
            return
        
        # Check if zoom changed and update object overlay if needed
        if self.show_objects.get():
            axes = self.figure.get_axes()
            if axes:
                ax = axes[0]
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                
                # Check if view changed
                if xlim != self._saved_xlim or ylim != self._saved_ylim:
                    self._saved_xlim = xlim
                    self._saved_ylim = ylim
                    # Update object overlay for new view
                    self._update_object_overlay_fast()

    def _update_pixel_display(self, event) -> None:
        """Update the pixel value display with current hover information."""
        try:
            col, row = int(event.xdata + 0.5), int(event.ydata + 0.5)
            H, W = self._current_data.data.field.shape
            
            if 0 <= row < H and 0 <= col < W:
                self.pixel_coords_label.config(text=f"({col},{row})")
                
                pixel_idx = row * W + col
                Y = self._current_data.data.Y
                pixel_values = Y[:, pixel_idx]
                
                values_text = self._format_channel_values(pixel_values)
                self.pixel_values_label.config(text=f"Values: {values_text}")
            else:
                self.pixel_coords_label.config(text="(-,-)")
                self.pixel_values_label.config(text="Values: -")
        except:
            self.pixel_coords_label.config(text="(-,-)")
            self.pixel_values_label.config(text="Values: -")

    def _format_value(self, value: float) -> str:
        """Format a single value with appropriate precision and notation."""
        if abs(value) < 1e-3 and value != 0:
            return f"{value:.2e}"
        elif abs(value) > 1e4:
            return f"{value:.2e}"
        elif abs(value) < 0.1 and value != 0:
            return f"{value:.4f}"
        elif abs(value) < 10:
            return f"{value:.2f}"
        else:
            return f"{value:.1f}"
    
    def _format_channel_values(self, pixel_values: np.ndarray) -> str:
        """Format channel values with smart display based on number of channels."""
        n_channels = len(pixel_values)
        
        if n_channels <= 4:
            formatted_values = [self._format_value(v) for v in pixel_values]
            return f"[{', '.join(formatted_values)}]"
        elif n_channels <= 8:
            first_three = [self._format_value(v) for v in pixel_values[:3]]
            remaining = n_channels - 3
            return f"[{', '.join(first_three)}, +{remaining} more]"
        else:
            min_val, max_val = np.min(pixel_values), np.max(pixel_values)
            mean_val = np.mean(pixel_values)
            return f"{n_channels}ch: μ={self._format_value(mean_val)} [{self._format_value(min_val)}-{self._format_value(max_val)}]"
    
    # ------------------------------------------------------------------
    # Object Overlay & Selection (OPTIMIZED)
    # ------------------------------------------------------------------
    
    def set_object_selection_callback(self, callback: callable) -> None:
        """Set callback for when objects are selected."""
        self.on_object_selection_changed = callback
    
    def _toggle_object_overlay(self) -> None:
        """Toggle object position overlay."""
        if self.show_objects.get():
            self._draw_object_overlay_fast()
        else:
            self._clear_object_overlay()
        self.canvas.draw_idle()
    
    def _get_visible_objects(self) -> list:
        """Return only objects within current viewport (FRUSTUM CULLING)."""
        if not self._current_data or not self._current_data.data.has_data:
            return []
        
        axes = self.figure.get_axes()
        if not axes:
            return []
        
        ax = axes[0]
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        all_objects = self._current_data.data.metadata.get('objects', [])
        
        # Add margin for objects near edges
        margin = 5.0
        visible = []
        for obj in all_objects:
            y, x = obj['position']
            if (xlim[0] - margin) <= x <= (xlim[1] + margin) and (ylim[1] - margin) <= y <= (ylim[0] + margin):
                visible.append(obj)
        
        return visible
    
    def _calculate_zoom_level(self) -> str:
        """Calculate current zoom level for adaptive rendering."""
        if not self._original_xlim or not self._saved_xlim:
            return 'full'
        
        axes = self.figure.get_axes()
        if not axes:
            return 'full'
        
        ax = axes[0]
        xlim = ax.get_xlim()
        
        # Calculate zoom factor
        original_width = self._original_xlim[1] - self._original_xlim[0]
        current_width = xlim[1] - xlim[0]
        zoom_factor = original_width / current_width
        
        if zoom_factor < 1.5:
            return 'far'  # Zoomed out or full view
        elif zoom_factor < 5:
            return 'medium'
        else:
            return 'near'  # Zoomed in
    
    def _draw_object_overlay_fast(self) -> None:
        """Draw object positions using fast PathCollection (OPTIMIZED)."""
        self._clear_object_overlay()
        
        if not self._current_data or not self._current_data.data.has_data:
            return
        
        # Get only visible objects (frustum culling)
        visible_objects = self._get_visible_objects()
        if not visible_objects:
            return
        
        axes = self.figure.get_axes()
        if not axes:
            return
        
        ax = axes[0]
        
        # Determine marker size based on zoom level (adaptive rendering)
        zoom_level = self._calculate_zoom_level()
        if zoom_level == 'far':
            markersize = 2  # Smaller dots when zoomed out
            linewidth = 0.5
        elif zoom_level == 'medium':
            markersize = 4
            linewidth = 1.0
        else:
            markersize = 6
            linewidth = 1.5
        
        # Prepare positions and colors
        positions = []
        colors = []
        sizes = []
        
        for obj in visible_objects:
            pos = obj.get('position', (0, 0))
            y, x = pos
            positions.append([x, y])
            
            obj_id = obj['id']
            if obj_id in self.selected_object_ids:
                colors.append('yellow')
                sizes.append(markersize * 1.5)  # Selected objects slightly larger
            else:
                colors.append('cyan')
                sizes.append(markersize)
        
        if not positions:
            return
        
        positions = np.array(positions)
        
        # Create single PathCollection for all objects (FAST!)
        self.object_collection = ax.scatter(
            positions[:, 0], positions[:, 1],
            c=colors, s=sizes, marker='o',
            facecolors='none', edgecolors=colors,
            linewidths=linewidth, picker=True
        )
    
    def _update_object_overlay_fast(self) -> None:
        """Update object overlay without full redraw (for zoom/pan)."""
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
        self._draw_object_overlay_fast()
    
    def _clear_object_overlay(self) -> None:
        """Remove object markers from display."""
        if self.object_collection:
            try:
                self.object_collection.remove()
            except ValueError:
                pass
            self.object_collection = None
    
    def _find_nearest_object(self, x: float, y: float, max_distance: float = 10.0) -> Optional[int]:
        """Find the nearest object to clicked position.
        
        Args:
            x, y: Click position in data coordinates
            max_distance: Maximum distance in pixels to consider
            
        Returns:
            Object ID if found within max_distance, None otherwise
        """
        if not self._current_data or not self._current_data.data.has_data:
            return None
        
        # Only search visible objects for performance
        visible_objects = self._get_visible_objects()
        if not visible_objects:
            return None
        
        min_dist = float('inf')
        nearest_id = None
        
        for obj in visible_objects:
            pos = obj.get('position', (0, 0))
            obj_y, obj_x = pos
            
            # Calculate distance
            dist = np.sqrt((x - obj_x)**2 + (y - obj_y)**2)
            
            if dist < min_dist and dist <= max_distance:
                min_dist = dist
                nearest_id = obj['id']
        
        return nearest_id
    
    def select_objects(self, object_ids: list) -> None:
        """Set selected objects and update display.
        
        Args:
            object_ids: List of object IDs to select
        """
        self.selected_object_ids = object_ids
        
        if self.show_objects.get():
            self._update_object_overlay_fast()
            self.canvas.draw_idle()
        
        # Notify callback
        if self.on_object_selection_changed:
            self.on_object_selection_changed(object_ids)
    
    def get_selected_objects(self) -> list:
        """Get currently selected object IDs."""
        return self.selected_object_ids.copy()
    
    # ------------------------------------------------------------------
    # Area Selection for Objects
    # ------------------------------------------------------------------
    
    def _toggle_area_select_mode(self) -> None:
        """Toggle area selection mode for objects."""
        self.object_area_select_mode = not self.object_area_select_mode
        
        if self.object_area_select_mode:
            self.area_select_btn.config(relief=tk.SUNKEN, bg='lightgreen')
            # Auto-enable object overlay
            if not self.show_objects.get():
                self.show_objects.set(True)
                self._toggle_object_overlay()
            # Deactivate spectral modes
            self._set_spectral_mode(SpectralMode.NONE)
        else:
            self.area_select_btn.config(relief=tk.RAISED, bg='SystemButtonFace')
            self._clear_area_selection_rect()
    
    def _start_area_selection(self, event) -> None:
        """Start area selection rectangle."""
        if not event.inaxes:
            return
        
        self.area_select_start = (event.xdata, event.ydata)
        self._clear_area_selection_rect()
    
    def _update_area_selection(self, event) -> None:
        """Update area selection rectangle during drag."""
        if not self.area_select_start or not event.inaxes:
            return
        
        axes = self.figure.get_axes()
        if not axes:
            return
        
        ax = axes[0]
        
        # Clear previous rectangle
        self._clear_area_selection_rect()
        
        # Draw new rectangle
        x0, y0 = self.area_select_start
        x1, y1 = event.xdata, event.ydata
        
        width = x1 - x0
        height = y1 - y0
        
        from matplotlib.patches import Rectangle
        self.area_select_rect = Rectangle((x0, y0), width, height,
                                          linewidth=2, edgecolor='yellow',
                                          facecolor='yellow', alpha=0.2)
        ax.add_patch(self.area_select_rect)
        self.canvas.draw_idle()
    
    def _complete_area_selection(self, event) -> None:
        """Complete area selection and select objects within rectangle."""
        if not self.area_select_start or not event.inaxes:
            self._clear_area_selection_rect()
            self.area_select_start = None
            return
        
        x0, y0 = self.area_select_start
        x1, y1 = event.xdata, event.ydata
        
        # Ensure x0 < x1 and y0 < y1
        xmin, xmax = min(x0, x1), max(x0, x1)
        ymin, ymax = min(y0, y1), max(y0, y1)
        
        # Find all objects within rectangle
        if self._current_data and self._current_data.data.has_data:
            objects = self._current_data.data.metadata.get('objects', [])
            selected_ids = []
            
            for obj in objects:
                y, x = obj['position']
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    selected_ids.append(obj['id'])
            
            # Handle multi-select with Ctrl
            if event.key in ('control', 'cmd'):
                # Add to existing selection
                for obj_id in selected_ids:
                    if obj_id not in self.selected_object_ids:
                        self.selected_object_ids.append(obj_id)
            else:
                # Replace selection
                self.selected_object_ids = selected_ids
            
            self.select_objects(self.selected_object_ids)
        
        # Clear rectangle and reset
        self._clear_area_selection_rect()
        self.area_select_start = None
    
    def _clear_area_selection_rect(self) -> None:
        """Clear area selection rectangle."""
        if self.area_select_rect:
            try:
                self.area_select_rect.remove()
            except ValueError:
                pass
            self.area_select_rect = None

