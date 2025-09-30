"""Main composite view for RGB image display with spectral analysis tools."""

from __future__ import annotations
from typing import Iterable, Optional
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from ...state import PlaygroundState
from ..utils import wavelength_to_rgb_nm
from .enums import SpectralMode
from .visual_settings import VisualSettingsManager
from .spectral_analysis import SpectralAnalysisTools


class CompositeView:
    """Matplotlib-backed view for the composite image display."""

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
        self._rgb_cache: Optional[np.ndarray] = None
        self._current_data: Optional[PlaygroundState] = None
        self.spectral_mode = SpectralMode.NONE
        
        # Store axis limits for zoom/pan preservation (only when image dimensions unchanged)
        self._saved_xlim: Optional[tuple] = None
        self._saved_ylim: Optional[tuple] = None
        self._last_image_shape: Optional[tuple] = None  # Track image dimensions (H, W)
        
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

    def _create_toolbar(self, parent: tk.Widget) -> None:
        """Create toolbar with matplotlib tools and spectral tools."""
        toolbar_frame = tk.Frame(parent)
        toolbar_frame.grid(row=2, column=0, sticky=tk.EW, pady=(2, 0))
        
        # Standard matplotlib toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.pack(side=tk.LEFT)
        
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
        
        self.area_btn = tk.Button(spectral_frame, text="R", width=3, relief=tk.RAISED,
                                  command=lambda: self._set_spectral_mode(SpectralMode.AREA))
        self.area_btn.pack(side=tk.LEFT, padx=1)
        
        self.clear_btn = tk.Button(spectral_frame, text="X", width=3, relief=tk.RAISED,
                                   command=lambda: self._set_spectral_mode(SpectralMode.NONE))
        self.clear_btn.pack(side=tk.LEFT, padx=(5, 0))

    def _create_visual_settings_button(self, parent: tk.Frame) -> None:
        """Create visual settings button."""
        visual_separator = ttk.Separator(parent, orient='vertical')
        visual_separator.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5))
        
        self.visual_settings_btn = tk.Button(parent, text="V", width=3, relief=tk.RAISED,
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

    @property
    def latest_rgb(self) -> Optional[np.ndarray]:
        """Get the latest rendered RGB image."""
        return self._rgb_cache

    def update(self, state: PlaygroundState, active_channels: Iterable[bool]) -> None:
        """Update the composite image display.
        
        Args:
            state: Current playground state
            active_channels: Which channels are active for display
        """
        self._current_data = state
        data = state.data
        
        if not data.has_data:
            self.figure.clear()
            self.figure.suptitle('No dataset loaded', fontsize=12)
            self._saved_xlim = None
            self._saved_ylim = None
            self._last_image_shape = None
            self._rgb_cache = None
            self._clear_overlays()
            self.canvas.draw_idle()
            return

        # Check if image dimensions changed (new dataset)
        current_shape = data.field.shape
        image_dimensions_changed = (self._last_image_shape != current_shape)
        
        # Only save zoom if image dimensions haven't changed
        if not image_dimensions_changed:
            self._save_axis_limits()
        else:
            # New image size - reset zoom
            self._saved_xlim = None
            self._saved_ylim = None
            self._last_image_shape = current_shape
        
        # Clear and redraw
        self.figure.clear()
        self._rgb_cache = None
        self._clear_overlays()

        # Render composite image
        rgb = self._render_composite(data, list(active_channels))
        self._rgb_cache = rgb

        # Display image
        ax = self.figure.add_subplot(1, 1, 1)
        ax.imshow(rgb, aspect='equal')
        ax.set_title('Composite Image', fontsize=14)
        ax.axis('off')
        ax.format_coord = lambda x, y: ''  # Disable matplotlib's coordinate display
        
        # Restore zoom/pan state only if dimensions unchanged
        if not image_dimensions_changed:
            self._restore_axis_limits(ax)

        self.canvas.draw_idle()

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
        try:
            axes = self.figure.get_axes()
            if axes:
                ax = axes[0]
                self._saved_xlim = ax.get_xlim()
                self._saved_ylim = ax.get_ylim()
        except:
            pass
    
    def _restore_axis_limits(self, ax) -> None:
        """Restore saved axis limits (zoom/pan state)."""
        try:
            if self._saved_xlim is not None and self._saved_ylim is not None:
                ax.set_xlim(self._saved_xlim)
                ax.set_ylim(self._saved_ylim)
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
            
            self.update(self._current_data, active_channels)
        except:
            if self._current_data and self._current_data.data.has_data:
                active_channels = [True] * self._current_data.data.Y.shape[0]
                self.update(self._current_data, active_channels)

    # ------------------------------------------------------------------
    # Spectral Mode & Mouse Event Handling
    # ------------------------------------------------------------------
    
    def _set_spectral_mode(self, mode: SpectralMode) -> None:
        """Set the current spectral analysis mode and update button states."""
        self.spectral_mode = mode
        self.spectral_tools.reset_state()
        self._clear_overlays()
        
        # Deactivate matplotlib toolbar tools
        if hasattr(self.toolbar, '_active') and self.toolbar._active:
            self.toolbar._active = None
        try:
            self.toolbar.home()
        except:
            pass
        
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
        """Clear any drawn overlays on the image."""
        overlay = self.spectral_tools.get_current_overlay()
        if overlay:
            try:
                overlay.remove()
            except ValueError:
                pass
            self.spectral_tools.clear_overlay()
            self.canvas.draw_idle()
    
    def _on_mouse_press(self, event) -> None:
        """Handle mouse press events for spectral analysis."""
        if event.button != 1:
            return
        if not event.inaxes or not self._current_data or not self._current_data.data.has_data:
            return
        
        if self.spectral_mode == SpectralMode.PIXEL:
            self.spectral_tools.handle_pixel_click(event)
        elif self.spectral_mode == SpectralMode.LINE:
            self._clear_overlays()
            self.spectral_tools.start_line(event)
        elif self.spectral_mode == SpectralMode.AREA:
            self._clear_overlays()
            self.spectral_tools.start_area(event)
    
    def _on_mouse_release(self, event) -> None:
        """Handle mouse release events for spectral analysis."""
        if event.button != 1:
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
