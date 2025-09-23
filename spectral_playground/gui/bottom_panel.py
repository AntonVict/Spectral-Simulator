from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Iterable, List

from .state import PlaygroundState
from .views.spectral import SpectralPanel
from .views.abundance import AbundancePanel
from .views.output import OutputPanel


class BottomPanel(ttk.Frame):
    """Container for spectral plots, abundance view, and log output."""

    def __init__(
        self,
        parent: tk.Widget,
        state: PlaygroundState,
        on_fluor_selection_changed: Callable[[], None],
    ) -> None:
        super().__init__(parent)
        self.state = state
        self.on_fluor_selection_changed = on_fluor_selection_changed

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)

        spectral_frame = ttk.LabelFrame(self, text='Spectral Profiles')
        spectral_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 4))

        spectral_controls = ttk.Frame(spectral_frame)
        spectral_controls.pack(fill=tk.X, padx=4, pady=2)

        ttk.Label(spectral_controls, text='Show:').pack(side=tk.LEFT)
        self._fluor_checks_container = ttk.Frame(spectral_controls)
        self._fluor_checks_container.pack(side=tk.LEFT, padx=(4, 0))
        self._spectral_expand_btn = ttk.Button(spectral_controls, text='Expand', command=lambda: None)
        self._spectral_expand_btn.pack(side=tk.RIGHT)

        self.spectral_panel = SpectralPanel(spectral_frame)
        self.fluor_vars: List[tk.BooleanVar] = []

        abundance_frame = ttk.LabelFrame(self, text='Abundance')
        abundance_frame.grid(row=0, column=1, sticky='nsew', padx=4)
        self.abundance_panel = AbundancePanel(abundance_frame)
        self.abundance_panel.pack(fill=tk.BOTH, expand=True)

        self.output_panel = OutputPanel(self)
        self.output_panel.grid(row=0, column=2, sticky='nsew', padx=(4, 0))

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
