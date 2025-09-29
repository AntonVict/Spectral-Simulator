from __future__ import annotations

from typing import Iterable, Optional
from enum import Enum

import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from ..state import PlaygroundState
from .utils import wavelength_to_rgb_nm


class SpectralMode(Enum):
    """Different spectral analysis modes."""
    NONE = "none"
    PIXEL = "pixel"
    LINE = "line"
    AREA = "area"


class CompositeView:
    """Matplotlib-backed view for the composite image display."""

    def __init__(self, parent: tk.Widget, on_visual_settings_changed: Optional[callable] = None) -> None:
        # Make the figure size more modest to fit better
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.figure.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.06)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas_widget = self.canvas.get_tk_widget()
        # Use grid to match parent's geometry manager
        self.canvas_widget.grid(row=1, column=0, sticky=tk.NSEW)
        
        # Configure parent grid weights to ensure proper expansion
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        # Create toolbar frame with matplotlib tools and spectral tools
        toolbar_frame = tk.Frame(parent)
        toolbar_frame.grid(row=2, column=0, sticky=tk.EW, pady=(2, 0))
        
        # Standard matplotlib toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.pack(side=tk.LEFT)
        
        # Disable the matplotlib toolbar's coordinate display
        try:
            self.toolbar.set_message = lambda s: None  # Disable status messages
        except:
            pass
        
        # Separator and spectral tools
        separator = ttk.Separator(toolbar_frame, orient='vertical')
        separator.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 5))
        
        spectral_frame = tk.Frame(toolbar_frame)
        spectral_frame.pack(side=tk.LEFT)
        
        tk.Label(spectral_frame, text="Spectral:", font=('TkDefaultFont', 8)).pack(side=tk.LEFT, padx=(0, 5))
        
        self.spectral_mode = SpectralMode.NONE
        self.mode_var = tk.StringVar(value=self.spectral_mode.value)
        
        # Spectral tool buttons
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

        # Visual settings button
        visual_separator = ttk.Separator(toolbar_frame, orient='vertical')
        visual_separator.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5))
        
        self.visual_settings_btn = tk.Button(toolbar_frame, text="🎨", width=3, relief=tk.RAISED,
                                           command=self._show_visual_settings)
        self.visual_settings_btn.pack(side=tk.LEFT, padx=(0, 10))

        # Visual settings state
        self.normalization_mode = tk.StringVar(value="per_channel")  # "per_channel" or "global"
        self._on_visual_settings_changed_callback = on_visual_settings_changed
        
        # Pixel value display
        self.pixel_info_frame = tk.Frame(toolbar_frame)
        self.pixel_info_frame.pack(side=tk.RIGHT, padx=(10, 0))
        
        tk.Label(self.pixel_info_frame, text="Pixel:", font=('TkDefaultFont', 8)).pack(side=tk.LEFT)
        self.pixel_coords_label = tk.Label(self.pixel_info_frame, text="(-,-)", 
                                          font=('TkDefaultFont', 8), width=8)
        self.pixel_coords_label.pack(side=tk.LEFT, padx=(2, 5))
        
        self.pixel_values_label = tk.Label(self.pixel_info_frame, text="Values: -", 
                                          font=('TkDefaultFont', 8), width=35)
        self.pixel_values_label.pack(side=tk.LEFT)

        # Spectral analysis state
        self._rgb_cache: Optional[np.ndarray] = None
        self._current_data: Optional['PlaygroundState'] = None
        self._line_start: Optional[tuple] = None
        self._area_start: Optional[tuple] = None
        self._current_overlay = None
        
        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_motion)

    @property
    def latest_rgb(self) -> Optional[np.ndarray]:
        return self._rgb_cache

    def update(self, state: PlaygroundState, active_channels: Iterable[bool]) -> None:
        self.figure.clear()
        self._rgb_cache = None
        self._current_data = state
        self._clear_overlays()

        data = state.data
        if not data.has_data:
            self.figure.suptitle('No dataset loaded', fontsize=12)
            self.canvas.draw_idle()
            return

        H, W = data.field.shape
        Y = data.Y
        channels = list(active_channels)
        if not any(channels):
            channels = [True] * Y.shape[0]

        rgb = np.zeros((H, W, 3), dtype=np.float32)
        eps = 1e-6
        
        # Calculate normalization scale based on mode
        if self.normalization_mode.get() == "global":
            # Global normalization: find maximum across all active channels
            global_max = 0.0
            for idx, flag in enumerate(channels):
                if flag:
                    channel_image = Y[idx].reshape(H, W)
                    channel_max = np.percentile(channel_image, 99.0)
                    global_max = max(global_max, channel_max)
            global_scale = global_max + eps
        
        for idx, flag in enumerate(channels):
            if not flag:
                continue
            channel_image = Y[idx].reshape(H, W)
            
            # Use appropriate scale based on normalization mode
            if self.normalization_mode.get() == "global":
                scale = global_scale
            else:  # per_channel (default)
                scale = np.percentile(channel_image, 99.0) + eps
            
            normalized = np.clip(channel_image / scale, 0.0, 1.0)
            color = np.array(wavelength_to_rgb_nm(data.spectral.channels[idx].center_nm), dtype=np.float32)
            rgb += normalized[..., None] * color[None, None, :]

        rgb = np.clip(rgb, 0.0, 1.0)
        self._rgb_cache = rgb

        ax = self.figure.add_subplot(1, 1, 1)
        ax.imshow(rgb, aspect='equal')
        ax.set_title('Composite Image', fontsize=14)
        ax.axis('off')
        
        # Disable matplotlib's coordinate display to avoid duplicate pixel info
        ax.format_coord = lambda x, y: ''

        self.canvas.draw_idle()

    def show_expanded(self, parent: tk.Tk) -> None:
        if self._rgb_cache is None:
            return
        window = tk.Toplevel(parent)
        window.title('Composite Image (Expanded)')
        window.geometry('1200x900')

        # Create main frame for better layout control
        main_frame = tk.Frame(window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        figure = Figure(figsize=(12, 8), dpi=100)
        figure.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05)
        canvas = FigureCanvasTkAgg(figure, master=main_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Create toolbar at bottom
        toolbar_frame = tk.Frame(main_frame)
        toolbar_frame.pack(fill=tk.X, pady=(5, 0))
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

        ax = figure.add_subplot(1, 1, 1)
        ax.imshow(self._rgb_cache, aspect='equal')
        ax.set_title('Composite Image (Expanded View)', fontsize=16)
        ax.axis('off')
        
        # Disable coordinate display in expanded view too
        ax.format_coord = lambda x, y: ''

        canvas.draw()

    # ------------------------------------------------------------------
    # Visual Settings Methods
    # ------------------------------------------------------------------
    
    def _show_visual_settings(self) -> None:
        """Show visual settings popup window."""
        window = tk.Toplevel()
        window.title('Visual Settings')
        window.geometry('400x300')
        window.transient()
        window.grab_set()
        
        # Make window not resizable for now
        window.resizable(False, False)
        
        main_frame = tk.Frame(window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Channel Normalization Section
        norm_frame = ttk.LabelFrame(main_frame, text="Channel Normalization", padding=10)
        norm_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(norm_frame, text="Controls how channel intensities are normalized for display:", 
                font=('TkDefaultFont', 9)).pack(anchor='w', pady=(0, 10))
        
        ttk.Radiobutton(norm_frame, text="Per-channel (current behavior)", 
                       variable=self.normalization_mode, value="per_channel",
                       command=self._on_visual_settings_changed).pack(anchor='w', pady=2)
        
        tk.Label(norm_frame, text="   • Each channel normalized to its own maximum", 
                font=('TkDefaultFont', 8), fg='gray').pack(anchor='w')
        tk.Label(norm_frame, text="   • Weak channels can appear artificially bright", 
                font=('TkDefaultFont', 8), fg='gray').pack(anchor='w', pady=(0, 8))
        
        ttk.Radiobutton(norm_frame, text="Global normalization", 
                       variable=self.normalization_mode, value="global",
                       command=self._on_visual_settings_changed).pack(anchor='w', pady=2)
        
        tk.Label(norm_frame, text="   • All channels normalized to strongest overall signal", 
                font=('TkDefaultFont', 8), fg='gray').pack(anchor='w')
        tk.Label(norm_frame, text="   • True relative channel strengths preserved", 
                font=('TkDefaultFont', 8), fg='gray').pack(anchor='w')
        
        # Future settings placeholder
        future_frame = ttk.LabelFrame(main_frame, text="Additional Settings", padding=10)
        future_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(future_frame, text="More visual controls will be added here...", 
                font=('TkDefaultFont', 9), fg='gray').pack(anchor='w')
        
        # Buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Close", command=window.destroy).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="Reset to Defaults", 
                  command=self._reset_visual_settings).pack(side=tk.RIGHT, padx=(0, 10))
    
    def _on_visual_settings_changed(self) -> None:
        """Called when visual settings change - trigger composite redraw."""
        if self._on_visual_settings_changed_callback:
            # Use the callback provided by the parent viewer
            self._on_visual_settings_changed_callback()
        else:
            # Fallback: try to redraw directly
            self._redraw_composite()
    
    def _reset_visual_settings(self) -> None:
        """Reset visual settings to defaults."""
        self.normalization_mode.set("per_channel")
        self._on_visual_settings_changed()
    
    def _redraw_composite(self) -> None:
        """Redraw the composite image with current settings (fallback method)."""
        if not self._current_data or not self._current_data.data.has_data:
            return
        
        # Try to get active channels from parent viewer (fallback approach)
        try:
            # Try to get active channels from parent viewer
            parent_viewer = self.canvas_widget.master
            while parent_viewer and not hasattr(parent_viewer, 'active_channel_flags'):
                parent_viewer = parent_viewer.master
            
            if parent_viewer and hasattr(parent_viewer, 'active_channel_flags'):
                active_channels = parent_viewer.active_channel_flags()
            else:
                # Fallback: assume all channels active
                active_channels = [True] * self._current_data.data.Y.shape[0]
            
            self.update(self._current_data, active_channels)
        except:
            # If we can't get active channels, just use all
            if self._current_data and self._current_data.data.has_data:
                active_channels = [True] * self._current_data.data.Y.shape[0]
                self.update(self._current_data, active_channels)

    # ------------------------------------------------------------------
    # Spectral Analysis Methods
    # ------------------------------------------------------------------
    
    def _set_spectral_mode(self, mode: SpectralMode) -> None:
        """Set the current spectral analysis mode and update button states."""
        self.spectral_mode = mode
        self._reset_selection_state()
        
        # Deactivate any active matplotlib toolbar tools
        if hasattr(self.toolbar, '_active') and self.toolbar._active:
            self.toolbar._active = None
        # Force toolbar to home state
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
        if self._current_overlay:
            try:
                self._current_overlay.remove()
            except ValueError:
                pass  # Already removed
            self._current_overlay = None
            self.canvas.draw_idle()
    
    def _reset_selection_state(self) -> None:
        """Reset the selection state variables."""
        self._line_start = None
        self._area_start = None
        self._clear_overlays()
    
    def _on_mouse_press(self, event) -> None:
        """Handle mouse press events for spectral analysis."""
        # Only handle left mouse button (button 1)
        if event.button != 1:
            return
            
        if not event.inaxes or not self._current_data or not self._current_data.data.has_data:
            return
        
        if self.spectral_mode == SpectralMode.PIXEL:
            self._handle_pixel_click(event)
        elif self.spectral_mode == SpectralMode.LINE:
            self._handle_line_start(event)
        elif self.spectral_mode == SpectralMode.AREA:
            self._handle_area_start(event)
    
    def _on_mouse_release(self, event) -> None:
        """Handle mouse release events for spectral analysis."""
        # Only handle left mouse button (button 1)
        if event.button != 1:
            return
            
        if not event.inaxes or not self._current_data or not self._current_data.data.has_data:
            return
        
        if self.spectral_mode == SpectralMode.LINE and self._line_start:
            self._handle_line_end(event)
        elif self.spectral_mode == SpectralMode.AREA and self._area_start:
            self._handle_area_end(event)
    
    def _on_mouse_motion(self, event) -> None:
        """Handle mouse motion for dynamic overlays and pixel value display."""
        if not event.inaxes or not self._current_data or not self._current_data.data.has_data:
            self.pixel_coords_label.config(text="(-,-)")
            self.pixel_values_label.config(text="Values: -")
            return
        
        # Update pixel value display
        self._update_pixel_display(event)
        
        # Handle spectral mode overlays
        if self.spectral_mode == SpectralMode.LINE and self._line_start:
            self._update_line_preview(event)
        elif self.spectral_mode == SpectralMode.AREA and self._area_start:
            self._update_area_preview(event)
    
    def _handle_pixel_click(self, event) -> None:
        """Handle single pixel spectral analysis."""
        col, row = int(event.xdata + 0.5), int(event.ydata + 0.5)
        H, W = self._current_data.data.field.shape
        
        if 0 <= row < H and 0 <= col < W:
            pixel_idx = row * W + col
            self._show_pixel_spectrum(pixel_idx, (row, col))
    
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
            # Show all values
            formatted_values = [self._format_value(v) for v in pixel_values]
            return f"[{', '.join(formatted_values)}]"
        elif n_channels <= 8:
            # Show first 3, then summary
            first_three = [self._format_value(v) for v in pixel_values[:3]]
            remaining = n_channels - 3
            return f"[{', '.join(first_three)}, +{remaining} more]"
        else:
            # Show statistics for many channels
            min_val, max_val = np.min(pixel_values), np.max(pixel_values)
            mean_val = np.mean(pixel_values)
            return f"{n_channels}ch: μ={self._format_value(mean_val)} [{self._format_value(min_val)}-{self._format_value(max_val)}]"

    def _update_pixel_display(self, event) -> None:
        """Update the pixel value display with current hover information."""
        try:
            col, row = int(event.xdata + 0.5), int(event.ydata + 0.5)
            H, W = self._current_data.data.field.shape
            
            if 0 <= row < H and 0 <= col < W:
                # Update coordinates
                self.pixel_coords_label.config(text=f"({col},{row})")
                
                # Get pixel values from all channels
                pixel_idx = row * W + col
                Y = self._current_data.data.Y
                pixel_values = Y[:, pixel_idx]
                
                # Format values with smart display
                values_text = self._format_channel_values(pixel_values)
                self.pixel_values_label.config(text=f"Values: {values_text}")
            else:
                self.pixel_coords_label.config(text="(-,-)")
                self.pixel_values_label.config(text="Values: -")
        except:
            self.pixel_coords_label.config(text="(-,-)")
            self.pixel_values_label.config(text="Values: -")

    def _handle_line_start(self, event) -> None:
        """Start line profile selection."""
        self._clear_overlays()  # Clear any existing overlays first
        self._line_start = (event.xdata, event.ydata)
    
    def _handle_line_end(self, event) -> None:
        """Complete line profile analysis."""
        if self._line_start:
            end_pos = (event.xdata, event.ydata)
            self._show_line_profile(self._line_start, end_pos)
            self._line_start = None
            self._clear_overlays()
    
    def _handle_area_start(self, event) -> None:
        """Start area selection."""
        self._clear_overlays()  # Clear any existing overlays first
        self._area_start = (event.xdata, event.ydata)
    
    def _handle_area_end(self, event) -> None:
        """Complete area selection analysis."""
        if self._area_start:
            end_pos = (event.xdata, event.ydata)
            self._show_area_spectrum(self._area_start, end_pos)
            self._area_start = None
            self._clear_overlays()
    
    def _update_line_preview(self, event) -> None:
        """Show preview of line being drawn."""
        if self._line_start:
            self._clear_overlays()
            ax = self.figure.axes[0]
            line = Line2D([self._line_start[0], event.xdata], 
                         [self._line_start[1], event.ydata], 
                         color='red', linewidth=2, alpha=0.7)
            ax.add_line(line)
            self._current_overlay = line
            self.canvas.draw_idle()
    
    def _update_area_preview(self, event) -> None:
        """Show preview of area being selected."""
        if self._area_start:
            self._clear_overlays()
            ax = self.figure.axes[0]
            x0, y0 = self._area_start
            width = event.xdata - x0
            height = event.ydata - y0
            rect = Rectangle((x0, y0), width, height, 
                           linewidth=2, edgecolor='blue', facecolor='none', alpha=0.7)
            ax.add_patch(rect)
            self._current_overlay = rect
            self.canvas.draw_idle()
    
    def _show_pixel_spectrum(self, pixel_idx: int, coords: tuple) -> None:
        """Display spectral profile for a single pixel."""
        data = self._current_data.data
        Y = data.Y  # (L, P)
        spectral = data.spectral
        
        # Get pixel spectrum
        pixel_spectrum = Y[:, pixel_idx]  # (L,)
        channel_centers = np.array([ch.center_nm for ch in spectral.channels])
        
        # Create popup window
        window = tk.Toplevel()
        window.title(f'Pixel Spectrum - ({coords[0]}, {coords[1]})')
        window.geometry('800x600')
        
        # Create matplotlib figure
        fig = Figure(figsize=(8, 6), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ax = fig.add_subplot(1, 1, 1)
        
        # Plot channel data as bars
        ax.bar(channel_centers, pixel_spectrum, width=20, alpha=0.7, 
               color=[wavelength_to_rgb_nm(c) for c in channel_centers])
        
        # Interpolate to smooth curve
        if len(channel_centers) > 1:
            lambdas = spectral.lambdas
            smooth_spectrum = np.interp(lambdas, channel_centers, pixel_spectrum)
            ax.plot(lambdas, smooth_spectrum, 'k-', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Intensity')
        ax.set_title(f'Spectral Profile - Pixel ({coords[0]}, {coords[1]})')
        ax.grid(True, alpha=0.3)
        
        canvas.draw()
    
    def _show_line_profile(self, start: tuple, end: tuple) -> None:
        """Display spectral profiles along a line."""
        data = self._current_data.data
        Y = data.Y  # (L, P)
        H, W = data.field.shape
        spectral = data.spectral
        
        # Generate line coordinates
        x0, y0 = max(0, min(W-1, int(start[0] + 0.5))), max(0, min(H-1, int(start[1] + 0.5)))
        x1, y1 = max(0, min(W-1, int(end[0] + 0.5))), max(0, min(H-1, int(end[1] + 0.5)))
        
        # Get line pixels using Bresenham-like algorithm
        distance = max(abs(x1 - x0), abs(y1 - y0))
        if distance == 0:
            return
        
        x_coords = np.linspace(x0, x1, distance + 1, dtype=int)
        y_coords = np.linspace(y0, y1, distance + 1, dtype=int)
        
        # Extract spectra along line
        line_spectra = []
        distances = []
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            pixel_idx = y * W + x
            line_spectra.append(Y[:, pixel_idx])
            distances.append(i)
        
        line_spectra = np.array(line_spectra)  # (distance, L)
        
        # Create popup window
        window = tk.Toplevel()
        window.title('Line Profile Spectral Analysis')
        window.geometry('1000x750')
        
        # Create main frame
        main_frame = tk.Frame(window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create control frame at top
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Add toggle for total intensity
        show_total_var = tk.BooleanVar(value=False)  # Off by default
        total_intensity_check = tk.Checkbutton(control_frame, 
                                              text="Show total intensity (sum of all channels)",
                                              variable=show_total_var,
                                              command=lambda: self._update_line_plot(fig, canvas, line_spectra, distances, spectral, show_total_var.get()))
        total_intensity_check.pack(side=tk.LEFT)
        
        # Create matplotlib figure
        fig = Figure(figsize=(10, 7), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=main_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial plot (without total intensity by default)
        self._update_line_plot(fig, canvas, line_spectra, distances, spectral, show_total_var.get())
    
    def _update_line_plot(self, fig, canvas, line_spectra, distances, spectral, show_total=False):
        """Update the line profile plot with or without total intensity."""
        # Clear the figure
        fig.clear()
        
        # Create subplots
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        
        # Plot 1: Intensity vs distance for each channel
        channel_centers = np.array([ch.center_nm for ch in spectral.channels])
        
        # Plot individual channels
        for i, center in enumerate(channel_centers):
            color = wavelength_to_rgb_nm(center)
            alpha = 0.7 if show_total else 1.0  # More transparent if total is shown
            ax1.plot(distances, line_spectra[:, i], color=color, 
                    label=f'Ch{i+1} ({center:.0f}nm)', linewidth=2, alpha=alpha)
        
        # Configure primary axis
        ax1.set_xlabel('Distance (pixels)')
        ax1.set_ylabel('Channel Intensity', color='darkblue')
        title = 'Intensity vs Distance Along Line'
        if show_total:
            title += ' (with Total)'
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='y', labelcolor='darkblue')
        
        if show_total:
            # Calculate total intensity (sum of all channels)
            total_intensity = np.sum(line_spectra, axis=1)
            
            # Create secondary y-axis for total intensity
            ax1_twin = ax1.twinx()
            ax1_twin.plot(distances, total_intensity, color='black', linewidth=3, 
                         label='Total Intensity', linestyle='--', alpha=0.9)
            
            # Configure secondary axis
            ax1_twin.set_ylabel('Total Intensity (Sum)', color='black')
            ax1_twin.tick_params(axis='y', labelcolor='black')
            
            # Combine legends from both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, 
                      bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        else:
            # Just the regular legend for channels
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
        # Plot 2: 2D spectral map
        im = ax2.imshow(line_spectra.T, aspect='auto', cmap='viridis', 
                       extent=[0, len(distances)-1, channel_centers[0], channel_centers[-1]])
        ax2.set_xlabel('Distance (pixels)')
        ax2.set_ylabel('Wavelength (nm)')
        ax2.set_title('Spectral Intensity Map Along Line')
        fig.colorbar(im, ax=ax2, label='Intensity')
        
        fig.tight_layout()
        canvas.draw()
    
    def _show_area_spectrum(self, start: tuple, end: tuple) -> None:
        """Display average spectral profile for selected area."""
        data = self._current_data.data
        Y = data.Y  # (L, P)
        H, W = data.field.shape
        spectral = data.spectral
        
        # Define area bounds
        x0, y0 = int(min(start[0], end[0]) + 0.5), int(min(start[1], end[1]) + 0.5)
        x1, y1 = int(max(start[0], end[0]) + 0.5), int(max(start[1], end[1]) + 0.5)
        
        # Clamp to image bounds
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W-1, x1), min(H-1, y1)
        
        if x0 >= x1 or y0 >= y1:
            return
        
        # Extract spectra from area
        area_spectra = []
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                pixel_idx = y * W + x
                area_spectra.append(Y[:, pixel_idx])
        
        area_spectra = np.array(area_spectra)  # (pixels, L)
        mean_spectrum = np.mean(area_spectra, axis=0)
        std_spectrum = np.std(area_spectra, axis=0)
        
        # Create popup window
        window = tk.Toplevel()
        window.title(f'Area Spectrum - ({x1-x0+1}×{y1-y0+1} pixels)')
        window.geometry('800x600')
        
        fig = Figure(figsize=(8, 6), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ax = fig.add_subplot(1, 1, 1)
        
        channel_centers = np.array([ch.center_nm for ch in spectral.channels])
        
        # Plot mean with error bars
        colors = [wavelength_to_rgb_nm(c) for c in channel_centers]
        ax.bar(channel_centers, mean_spectrum, width=20, alpha=0.7, color=colors,
               yerr=std_spectrum, capsize=5)
        
        # Interpolate smooth curve
        if len(channel_centers) > 1:
            lambdas = spectral.lambdas
            smooth_mean = np.interp(lambdas, channel_centers, mean_spectrum)
            ax.plot(lambdas, smooth_mean, 'k-', linewidth=2, alpha=0.8, label='Mean')
        
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Intensity')
        ax.set_title(f'Average Spectral Profile - Area ({x1-x0+1}×{y1-y0+1} pixels)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        canvas.draw()
