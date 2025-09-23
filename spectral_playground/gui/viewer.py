from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk
from typing import Callable, Iterable, List

from .state import PlaygroundState
from .views.composite import CompositeView


class ViewerPanel(ttk.Frame):
    """Right-hand panel containing the composite view and controls."""

    def __init__(
        self,
        parent: tk.Widget,
        state: PlaygroundState,
        on_generate: Callable[[], None],
        on_load: Callable[[], None],
        on_save_dataset: Callable[[], None],
        on_save_composite: Callable[[], None],
        on_export_plots: Callable[[], None],
        on_open_folder: Callable[[], None],
        on_channels_changed: Callable[[], None],
        on_expand_composite: Callable[[], None],
    ) -> None:
        super().__init__(parent)
        self.state = state
        self.on_channels_changed = on_channels_changed

        self.columnconfigure(0, weight=1)
        main_frame = ttk.Frame(self)
        main_frame.grid(row=0, column=0, sticky='nsew', padx=8, pady=8)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=0)
        main_frame.rowconfigure(0, weight=1)

        image_frame = ttk.Frame(main_frame)
        image_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 8))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(1, weight=1)

        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=0, column=1, sticky='ns')

        actions_group = ttk.LabelFrame(controls_frame, text='Actions')
        actions_group.pack(fill=tk.X, padx=4, pady=4)

        ttk.Button(actions_group, text='Generate Data', command=on_generate, width=14).pack(pady=2)
        ttk.Button(actions_group, text='Load Dataset', command=on_load, width=14).pack(pady=2)
        ttk.Button(actions_group, text='Save Dataset', command=on_save_dataset, width=14).pack(pady=2)
        ttk.Button(actions_group, text='Save Composite', command=on_save_composite, width=14).pack(pady=2)
        ttk.Button(actions_group, text='Export Plots', command=on_export_plots, width=14).pack(pady=2)
        ttk.Button(actions_group, text='Open Save Folder', command=on_open_folder, width=14).pack(pady=(2, 0))

        channel_group = ttk.LabelFrame(controls_frame, text='Channels')
        channel_group.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.channel_checks_frame = ttk.Frame(channel_group)
        self.channel_checks_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.channel_vars: List[tk.BooleanVar] = []

        ttk.Button(channel_group, text='Select All', command=self.select_all_channels).pack(fill=tk.X, padx=4, pady=(4, 2))
        ttk.Button(channel_group, text='Select None', command=self.select_no_channels).pack(fill=tk.X, padx=4, pady=(0, 4))

        top_controls = ttk.Frame(image_frame)
        top_controls.grid(row=0, column=0, sticky='ew')
        ttk.Label(top_controls, text='Composite View').pack(side=tk.LEFT)
        ttk.Button(top_controls, text='Expand', command=on_expand_composite, width=12).pack(side=tk.RIGHT)

        self.composite_view = CompositeView(image_frame)

    def set_channels(self, names: Iterable[str]) -> None:
        for widget in self.channel_checks_frame.winfo_children():
            widget.destroy()
        self.channel_vars.clear()
        for idx, name in enumerate(names):
            var = tk.BooleanVar(value=True)
            chk = ttk.Checkbutton(
                self.channel_checks_frame,
                text=name,
                variable=var,
                command=self._channels_changed,
            )
            chk.pack(anchor='w', pady=1)
            self.channel_vars.append(var)
        self._channels_changed()

    def _channels_changed(self) -> None:
        flags = tuple(var.get() for var in self.channel_vars)
        self.state.selections.update_channels(flags)
        self.on_channels_changed()

    def select_all_channels(self) -> None:
        for var in self.channel_vars:
            var.set(True)
        self._channels_changed()

    def select_no_channels(self) -> None:
        for var in self.channel_vars:
            var.set(False)
        self._channels_changed()

    def active_channel_flags(self) -> List[bool]:
        return [var.get() for var in self.channel_vars]
