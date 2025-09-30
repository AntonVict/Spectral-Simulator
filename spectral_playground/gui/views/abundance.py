from __future__ import annotations

from typing import List, Optional

import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ..state import PlaygroundState


class AbundancePanel(ttk.Frame):
    """Displays abundance maps for individual fluorophores."""

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self._fluorophore_names = []  # Store fluorophore names
        
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=4, pady=2)

        ttk.Label(control_frame, text='Show:').pack(side=tk.LEFT)
        self._selected = tk.StringVar(value='')
        self.dropdown = ttk.Combobox(control_frame, textvariable=self._selected, state='readonly', width=10)
        self.dropdown.pack(side=tk.LEFT, padx=(4, 8))

        self.expand_button = ttk.Button(control_frame, text='Expand', command=lambda: None)
        self.expand_button.pack(side=tk.RIGHT)

        self.figure = Figure(figsize=(4, 3), dpi=80)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def set_expand_callback(self, callback) -> None:
        self.expand_button.config(command=callback)

    def set_fluorophores(self, names: List[str]) -> None:
        self._fluorophore_names = names  # Store for later use in plot titles
        self.dropdown['values'] = names
        if names:
            self._selected.set(names[0])
        else:
            self._selected.set('')

    def selected_index(self) -> Optional[int]:
        value = self._selected.get()
        if not value:
            return None
        values = list(self.dropdown['values'])
        if value in values:
            return values.index(value)
        if value.startswith('F') and value[1:].isdigit():
            return int(value[1:]) - 1
        return None

    def update(self, state: PlaygroundState) -> None:
        self.figure.clear()
        data = state.data
        ax = self.figure.add_subplot(1, 1, 1)

        if not data.has_data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=10)
            ax.axis('off')
            self.canvas.draw_idle()
            return

        idx = self.selected_index()
        if idx is None or idx >= data.A.shape[0]:
            ax.text(0.5, 0.5, 'Select a fluorophore', ha='center', va='center', fontsize=10)
            ax.axis('off')
            self.canvas.draw_idle()
            return

        H, W = data.field.shape
        abundance = data.A[idx].reshape(H, W)
        im = ax.imshow(abundance, cmap='magma')
        
        # Use actual fluorophore name instead of F{idx+1}
        fluor_name = self._fluorophore_names[idx] if idx < len(self._fluorophore_names) else f'F{idx + 1}'
        ax.set_title(f'Fluorophore {fluor_name}', fontsize=10)
        ax.axis('off')
        self.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        self.canvas.draw_idle()

    def show_expanded(self, parent: tk.Tk, state: PlaygroundState) -> None:
        data = state.data
        if not data.has_data:
            return

        window = tk.Toplevel(parent)
        window.title('Abundance Maps (Expanded)')
        window.geometry('1000x700')

        figure = Figure(figsize=(12, 8), dpi=100)
        canvas = FigureCanvasTkAgg(figure, master=window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        H, W = data.field.shape
        K = data.A.shape[0]
        if K <= 2:
            rows, cols = 1, K
        elif K <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 2, 3

        for idx in range(min(K, rows * cols)):
            ax = figure.add_subplot(rows, cols, idx + 1)
            im = ax.imshow(data.A[idx].reshape(H, W), cmap='magma')
            
            # Use actual fluorophore name instead of F{idx+1}
            fluor_name = self._fluorophore_names[idx] if idx < len(self._fluorophore_names) else f'F{idx + 1}'
            ax.set_title(fluor_name, fontsize=12)
            ax.axis('off')
            figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        figure.tight_layout()
        canvas.draw()
