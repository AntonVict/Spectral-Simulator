from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Iterable, List

from .state import PlaygroundState
from .views.spectral import SpectralPanel
from .views.abundance import AbundancePanel
from .views.output import OutputPanel
from .views.quick_inspector import QuickInspectorPanel


class BottomPanel(ttk.Frame):
    """Container for spectral plots, abundance view, and log output."""

    def __init__(
        self,
        parent: tk.Widget,
        state: PlaygroundState,
        on_fluor_selection_changed: Callable[[], None],
        on_open_full_inspector: Callable[[], None],
        get_data_callback: Callable,
        get_fluorophore_names_callback: Callable,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self.on_fluor_selection_changed = on_fluor_selection_changed

        # 4-column layout: Spectral | Abundance | Quick Inspector | Output
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.columnconfigure(3, weight=1)

        # Column 1: Spectral Profiles
        spectral_frame = ttk.LabelFrame(self, text='Spectral Profiles')
        spectral_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 4))

        spectral_controls = ttk.Frame(spectral_frame)
        spectral_controls.pack(fill=tk.X, padx=4, pady=2)

        ttk.Label(spectral_controls, text='Show:').pack(side=tk.LEFT)
        
        # Expand button on the right
        self._spectral_expand_btn = ttk.Button(spectral_controls, text='Expand', command=lambda: None)
        self._spectral_expand_btn.pack(side=tk.RIGHT, padx=(4, 0))
        
        # Create scrollable container for fluorophore checkboxes
        self._fluor_scroll_container = tk.Frame(spectral_controls, height=25)
        self._fluor_scroll_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 4))
        
        self._fluor_canvas = tk.Canvas(self._fluor_scroll_container, height=25, highlightthickness=0)
        self._fluor_scrollbar = ttk.Scrollbar(self._fluor_scroll_container, orient='horizontal', command=self._fluor_canvas.xview)
        self._fluor_checks_container = ttk.Frame(self._fluor_canvas)
        
        self._fluor_canvas.configure(xscrollcommand=self._fluor_scrollbar.set)
        self._fluor_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._fluor_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self._canvas_window = self._fluor_canvas.create_window((0, 0), window=self._fluor_checks_container, anchor='nw')
        self._fluor_checks_container.bind('<Configure>', self._on_fluor_container_configure)

        self.spectral_panel = SpectralPanel(spectral_frame)
        self.fluor_vars: List[tk.BooleanVar] = []
        self.measured_total_var = tk.BooleanVar(value=True)

        # Column 2: Abundance
        abundance_frame = ttk.LabelFrame(self, text='Abundance')
        abundance_frame.grid(row=0, column=1, sticky='nsew', padx=4)
        self.abundance_panel = AbundancePanel(abundance_frame)
        self.abundance_panel.pack(fill=tk.BOTH, expand=True)

        # Column 3: Quick Inspector
        quick_inspector_frame = ttk.LabelFrame(self, text='Quick Inspector')
        quick_inspector_frame.grid(row=0, column=2, sticky='nsew', padx=4)
        self.quick_inspector = QuickInspectorPanel(
            quick_inspector_frame,
            get_data_callback=get_data_callback,
            get_fluorophore_names_callback=get_fluorophore_names_callback,
            on_open_full_inspector=on_open_full_inspector
        )
        self.quick_inspector.pack(fill=tk.BOTH, expand=True)

        # Column 4: Logs/Output
        self.output_panel = OutputPanel(self)
        self.output_panel.grid(row=0, column=3, sticky='nsew', padx=(4, 0))

    def configure_expanders(self, spectral_cb: Callable[[], None], abundance_cb: Callable[[], None]) -> None:
        self._spectral_expand_btn.config(command=spectral_cb)
        self.abundance_panel.set_expand_callback(abundance_cb)

    def set_fluorophores(self, names: List[str]) -> None:
        for widget in self._fluor_checks_container.winfo_children():
            widget.destroy()
        self.fluor_vars.clear()
        self.abundance_panel.set_fluorophores(names)

        if not names:
            self.state.selections.update_fluors(tuple())
            return

        # Add "Measured Total" checkbox first
        measured_chk = ttk.Checkbutton(
            self._fluor_checks_container,
            text='Measured Total',
            variable=self.measured_total_var,
            command=self._measured_total_changed,
        )
        measured_chk.pack(side=tk.LEFT, padx=2)
        
        # Add separator
        ttk.Separator(self._fluor_checks_container, orient='vertical').pack(side=tk.LEFT, padx=4, fill=tk.Y, pady=2)

        for idx, name in enumerate(names):
            var = tk.BooleanVar(value=True)
            chk = ttk.Checkbutton(
                self._fluor_checks_container,
                text=name,
                variable=var,
                command=self._fluor_toggles_changed,
            )
            chk.pack(side=tk.LEFT, padx=2)
            self.fluor_vars.append(var)

        self._fluor_toggles_changed()
        self.abundance_panel.dropdown.unbind('<<ComboboxSelected>>')
        self.abundance_panel.dropdown.bind('<<ComboboxSelected>>', lambda _e: self.on_fluor_selection_changed())

    def active_fluor_flags(self) -> List[bool]:
        return [var.get() for var in self.fluor_vars]

    def update_views(self, active_channels: Iterable[bool]) -> None:
        self.spectral_panel.update(self.state, active_channels, self.active_fluor_flags())
        self.abundance_panel.update(self.state)

    def log(self, message: str) -> None:
        self.output_panel.log(message)

    def clear_log(self) -> None:
        self.output_panel.clear()

    def _fluor_toggles_changed(self) -> None:
        flags = tuple(var.get() for var in self.fluor_vars)
        self.state.selections.update_fluors(flags)
        self.on_fluor_selection_changed()
    
    def _measured_total_changed(self) -> None:
        self.state.selections.set_measured_total(self.measured_total_var.get())
        self.on_fluor_selection_changed()
    
    def _on_fluor_container_configure(self, event) -> None:
        # Update scrollable region when container size changes
        self._fluor_canvas.configure(scrollregion=self._fluor_canvas.bbox('all'))
