from __future__ import annotations

from typing import Iterable, List

import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ..state import PlaygroundState
from .utils import wavelength_to_rgb_nm


class SpectralPanel:
    """Displays spectral profiles and channel responses."""

    def __init__(self, parent: tk.Widget) -> None:
        self.figure = Figure(figsize=(6, 3), dpi=80)
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._last_channels: List[bool] = []
        self._last_fluors: List[bool] = []
        self._last_state: PlaygroundState | None = None

    def update(
        self,
        state: PlaygroundState,
        active_channels: Iterable[bool],
        active_fluors: Iterable[bool],
    ) -> None:
        self._last_state = state
        self._last_channels = list(active_channels)
        self._last_fluors = list(active_fluors)

        self.figure.clear()
        self._render(self.figure, state, self._last_channels, self._last_fluors)
        self.canvas.draw_idle()

    def show_expanded(self, parent: tk.Tk) -> None:
        if self._last_state is None:
            return
        window = tk.Toplevel(parent)
        window.title('Spectral Profiles (Expanded)')
        window.geometry('900x600')

        figure = Figure(figsize=(10, 6), dpi=100)
        canvas = FigureCanvasTkAgg(figure, master=window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self._render(figure, self._last_state, self._last_channels, self._last_fluors)
        canvas.draw()

    def _render(
        self,
        figure: Figure,
        state: PlaygroundState,
        active_channels: List[bool],
        active_fluors: List[bool],
    ) -> None:
        data = state.data
        if not data.has_data:
            ax = figure.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=10)
            ax.axis('off')
            return

        spectral = data.spectral
        lambdas = spectral.lambdas
        ax = figure.add_subplot(1, 1, 1)

        ax.set_xlim(float(lambdas[0]), float(lambdas[-1]))
        ax.set_ylim(0, 1.1)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Normalized Intensity')
        ax.set_title('Spectral Profiles', fontsize=10)

        for idx, ch in enumerate(spectral.channels):
            half = 0.5 * ch.bandwidth_nm
            low, high = ch.center_nm - half, ch.center_nm + half
            is_active = active_channels[idx] if idx < len(active_channels) else True
            if is_active:
                face = wavelength_to_rgb_nm(ch.center_nm)
                ax.axvspan(low, high, ymin=0, ymax=0.15, facecolor=face, alpha=0.7, edgecolor='k', linewidth=1)
                ax.text(ch.center_nm, 0.08, f'C{idx + 1}', ha='center', va='center', fontsize=8, weight='bold')
            else:
                ax.axvspan(low, high, ymin=0, ymax=0.15, facecolor='lightgray', alpha=0.3, edgecolor='gray')

        for idx, fluor in enumerate(spectral.fluors):
            if idx >= len(active_fluors) or not active_fluors[idx]:
                continue
            pdf = spectral._pdf(fluor)
            pdf_norm = pdf / (np.max(pdf) + 1e-9)
            peak_nm = float(lambdas[np.argmax(pdf)])
            color = wavelength_to_rgb_nm(peak_nm)
            ax.plot(lambdas, 0.2 + 0.6 * pdf_norm, color=color, linewidth=2, label=f'{fluor.name}')

        total_per_channel = np.sum(data.Y, axis=1)
        channel_centers = np.array([ch.center_nm for ch in spectral.channels])
        if len(channel_centers) >= 2:
            interp = np.interp(
                lambdas,
                channel_centers,
                total_per_channel,
                left=total_per_channel[0],
                right=total_per_channel[-1],
            )
        else:
            interp = np.full_like(lambdas, total_per_channel[0] if len(total_per_channel) else 0.0)

        curve = 0.2 + 0.6 * (interp / np.max(interp)) if np.max(interp) > 0 else np.zeros_like(interp) + 0.2
        ax.plot(lambdas, curve, color='black', linewidth=2, label='Measured total')

        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
