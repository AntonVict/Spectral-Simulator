"""Spectral analysis tools for composite view."""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import numpy as np
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ..utils import wavelength_to_rgb_nm

if TYPE_CHECKING:
    from ...state import PlaygroundState


class SpectralAnalysisTools:
    """Handles spectral analysis tools (pixel, line, area profiling)."""
    
    def __init__(self, state_getter: callable):
        """Initialize spectral analysis tools.
        
        Args:
            state_getter: Callable that returns current PlaygroundState
        """
        self.get_state = state_getter
        
        # Analysis state
        self._line_start: Optional[tuple] = None
        self._area_start: Optional[tuple] = None
        self._current_overlay = None
    
    def reset_state(self) -> None:
        """Reset the selection state variables."""
        self._line_start = None
        self._area_start = None
    
    def has_active_line(self) -> bool:
        """Check if line selection is in progress."""
        return self._line_start is not None
    
    def has_active_area(self) -> bool:
        """Check if area selection is in progress."""
        return self._area_start is not None
    
    def set_current_overlay(self, overlay) -> None:
        """Set the current overlay object."""
        self._current_overlay = overlay
    
    def get_current_overlay(self):
        """Get the current overlay object."""
        return self._current_overlay
    
    def clear_overlay(self) -> None:
        """Clear the current overlay reference."""
        self._current_overlay = None
    
    # Pixel Analysis
    # ------------------------------------------------------------------
    
    def handle_pixel_click(self, event) -> None:
        """Handle single pixel spectral analysis."""
        state = self.get_state()
        if not state or not state.data.has_data:
            return
        
        col, row = int(event.xdata + 0.5), int(event.ydata + 0.5)
        H, W = state.data.field.shape
        
        if 0 <= row < H and 0 <= col < W:
            pixel_idx = row * W + col
            self.show_pixel_spectrum(pixel_idx, (row, col))
    
    def show_pixel_spectrum(self, pixel_idx: int, coords: tuple) -> None:
        """Display spectral profile for a single pixel."""
        state = self.get_state()
        if not state or not state.data.has_data:
            return
        
        data = state.data
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
    
    # Line Analysis
    # ------------------------------------------------------------------
    
    def start_line(self, event) -> None:
        """Start line profile selection."""
        self._line_start = (event.xdata, event.ydata)
    
    def update_line_preview(self, event, ax, canvas) -> Optional[Line2D]:
        """Show preview of line being drawn.
        
        Returns:
            Line2D object for the overlay
        """
        if not self._line_start:
            return None
        
        line = Line2D([self._line_start[0], event.xdata], 
                     [self._line_start[1], event.ydata], 
                     color='red', linewidth=2, alpha=0.7)
        ax.add_line(line)
        return line
    
    def complete_line(self, event) -> None:
        """Complete line profile analysis."""
        if self._line_start:
            end_pos = (event.xdata, event.ydata)
            self.show_line_profile(self._line_start, end_pos)
            self._line_start = None
    
    def show_line_profile(self, start: tuple, end: tuple) -> None:
        """Display spectral profiles along a line."""
        state = self.get_state()
        if not state or not state.data.has_data:
            return
        
        data = state.data
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
    
    # Area Analysis
    # ------------------------------------------------------------------
    
    def start_area(self, event) -> None:
        """Start area selection."""
        self._area_start = (event.xdata, event.ydata)
    
    def update_area_preview(self, event, ax, canvas) -> Optional[Rectangle]:
        """Show preview of area being selected.
        
        Returns:
            Rectangle object for the overlay
        """
        if not self._area_start:
            return None
        
        x0, y0 = self._area_start
        width = event.xdata - x0
        height = event.ydata - y0
        rect = Rectangle((x0, y0), width, height, 
                       linewidth=2, edgecolor='blue', facecolor='none', alpha=0.7)
        ax.add_patch(rect)
        return rect
    
    def complete_area(self, event) -> None:
        """Complete area selection analysis."""
        if self._area_start:
            end_pos = (event.xdata, event.ydata)
            self.show_area_spectrum(self._area_start, end_pos)
            self._area_start = None
    
    def show_area_spectrum(self, start: tuple, end: tuple) -> None:
        """Display average spectral profile for selected area."""
        state = self.get_state()
        if not state or not state.data.has_data:
            return
        
        data = state.data
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
