"""Pixel value information display."""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import tkinter as tk

if TYPE_CHECKING:
    from ....state import PlaygroundState


class PixelInfoDisplay:
    """Displays pixel coordinate and value information."""
    
    def __init__(self, parent_frame: tk.Frame):
        """Initialize pixel info display.
        
        Args:
            parent_frame: Parent frame for the pixel info widgets
        """
        self.pixel_info_frame = tk.Frame(parent_frame)
        self.pixel_info_frame.pack(side=tk.RIGHT, padx=(10, 0))
        
        tk.Label(self.pixel_info_frame, text="Pixel:", font=('TkDefaultFont', 8)).pack(side=tk.LEFT)
        self.pixel_coords_label = tk.Label(self.pixel_info_frame, text="(-,-)", 
                                          font=('TkDefaultFont', 8), width=8)
        self.pixel_coords_label.pack(side=tk.LEFT, padx=(2, 5))
        
        self.pixel_values_label = tk.Label(self.pixel_info_frame, text="Values: -", 
                                          font=('TkDefaultFont', 8), width=35)
        self.pixel_values_label.pack(side=tk.LEFT)
    
    def update_from_event(self, event, state: 'PlaygroundState') -> None:
        """Update pixel display from mouse event.
        
        Args:
            event: Matplotlib mouse event
            state: Current playground state
        """
        if not event.inaxes or not state or not state.data.has_data:
            self.clear()
            return
        
        try:
            col, row = int(event.xdata + 0.5), int(event.ydata + 0.5)
            H, W = state.data.field.shape
            
            if 0 <= row < H and 0 <= col < W:
                self.pixel_coords_label.config(text=f"({col},{row})")
                
                pixel_idx = row * W + col
                Y = state.data.Y
                pixel_values = Y[:, pixel_idx]
                
                values_text = self._format_channel_values(pixel_values)
                self.pixel_values_label.config(text=f"Values: {values_text}")
            else:
                self.clear()
        except:
            self.clear()
    
    def clear(self) -> None:
        """Clear pixel information display."""
        self.pixel_coords_label.config(text="(-,-)")
        self.pixel_values_label.config(text="Values: -")
    
    def _format_value(self, value: float) -> str:
        """Format a single value with appropriate precision and notation.
        
        Args:
            value: Value to format
            
        Returns:
            Formatted string
        """
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
        """Format channel values with smart display based on number of channels.
        
        Args:
            pixel_values: Array of pixel values across channels
            
        Returns:
            Formatted string
        """
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

