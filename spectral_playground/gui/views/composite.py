from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from ..state import PlaygroundState
from .utils import wavelength_to_rgb_nm


class CompositeView:
    """Matplotlib-backed view for the composite image display."""

    def __init__(self, parent: tk.Widget) -> None:
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

        # Create toolbar at bottom with proper frame
        toolbar_frame = tk.Frame(parent)
        toolbar_frame.grid(row=2, column=0, sticky=tk.EW, pady=(2, 0))
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        self._rgb_cache: Optional[np.ndarray] = None

    @property
    def latest_rgb(self) -> Optional[np.ndarray]:
        return self._rgb_cache

    def update(self, state: PlaygroundState, active_channels: Iterable[bool]) -> None:
        self.figure.clear()
        self._rgb_cache = None

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
        for idx, flag in enumerate(channels):
            if not flag:
                continue
            channel_image = Y[idx].reshape(H, W)
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

        canvas.draw()
