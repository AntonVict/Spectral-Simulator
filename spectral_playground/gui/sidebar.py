from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

from spectral_playground.core.spectra import Fluorophore

from .data_manager import GenerationConfig
from .fluorophore_editor import FluorophoreListManager
from .objects import ObjectLayersManager
from .settings_panels import (
    WavelengthGridPanel,
    DetectionChannelsPanel,
    ImageDimensionsPanel,
    RandomSeedPanel,
)


class Sidebar(ttk.Frame):
    """Left-hand settings sidebar."""

    def __init__(self, parent: tk.Widget, log_callback) -> None:
        super().__init__(parent)
        self.columnconfigure(0, weight=1)

        toggle_frame = ttk.Frame(self)
        toggle_frame.grid(row=0, column=0, sticky='ew', padx=4, pady=4)
        self._toggle_button = ttk.Button(toggle_frame, text='Hide Settings', command=self._toggle_visibility)
        self._toggle_button.pack(side=tk.LEFT)

        self._settings_container = ttk.Frame(self)
        self._settings_container.grid(row=1, column=0, sticky='nsew')
        self.rowconfigure(1, weight=1)

        scroll_frame = ttk.Frame(self._settings_container)
        scroll_frame.pack(fill=tk.BOTH, expand=True)
        scroll_frame.grid_columnconfigure(0, weight=1)
        scroll_frame.grid_rowconfigure(0, weight=1)

        canvas = tk.Canvas(scroll_frame, width=350)
        scrollbar = ttk.Scrollbar(scroll_frame, orient='vertical', command=canvas.yview)
        canvas.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')

        self._settings_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=self._settings_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')

        canvas.bind('<MouseWheel>', _on_mousewheel)
        self._settings_frame.bind(
            '<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all'))
        )

        self._build_panels(log_callback)
        self._visible = True

    def _build_panels(self, log_callback) -> None:
        row = 0
        self.panels = {}

        grid_group = ttk.LabelFrame(self._settings_frame, text='Wavelength Grid')
        grid_group.grid(row=row, column=0, sticky='ew', padx=2, pady=2)
        grid_group.columnconfigure(0, weight=1)
        self.panels['grid'] = WavelengthGridPanel(grid_group)
        row += 1

        channel_group = ttk.LabelFrame(self._settings_frame, text='Detection Channels')
        channel_group.grid(row=row, column=0, sticky='ew', padx=2, pady=2)
        channel_group.columnconfigure(0, weight=1)
        self.panels['channels'] = DetectionChannelsPanel(
            channel_group, 
            log_callback,
            self._get_wavelength_range
        )
        row += 1

        fluor_group = ttk.LabelFrame(self._settings_frame, text='Fluorophores')
        fluor_group.grid(row=row, column=0, sticky='ew', padx=2, pady=2)
        fluor_group.columnconfigure(0, weight=1)
        fluor_frame = ttk.Frame(fluor_group)
        fluor_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.fluor_manager = FluorophoreListManager(fluor_frame, log_callback)
        row += 1

        dims_group = ttk.LabelFrame(self._settings_frame, text='Image Dimensions')
        dims_group.grid(row=row, column=0, sticky='ew', padx=2, pady=2)
        dims_group.columnconfigure(0, weight=1)
        self.panels['dimensions'] = ImageDimensionsPanel(dims_group)
        row += 1

        objects_group = ttk.LabelFrame(self._settings_frame, text='Objects')
        objects_group.grid(row=row, column=0, sticky='ew', padx=2, pady=2)
        objects_group.columnconfigure(0, weight=1)
        self.object_manager = ObjectLayersManager(objects_group, log_callback, self._get_image_dims, self._get_fluorophore_names)
        row += 1

        seed_group = ttk.LabelFrame(self._settings_frame, text='Random Seed')
        seed_group.grid(row=row, column=0, sticky='ew', padx=2, pady=2)
        seed_group.columnconfigure(0, weight=1)
        self.panels['seed'] = RandomSeedPanel(seed_group)

    def _get_image_dims(self):
        dims = self.panels['dimensions'].get_dimensions()
        return dims['H'], dims['W']
    
    def _get_wavelength_range(self):
        """Get current wavelength range for channel validation."""
        grid = self.panels['grid'].get_wavelength_grid()
        return (grid['start'], grid['stop'])
    
    def _get_fluorophore_names(self):
        """Get list of fluorophore names for the object layers dropdown."""
        try:
            fluorophores = self.fluor_manager.get_fluorophores()
            return [f.name for f in fluorophores]
        except:
            return ["F1", "F2", "F3"]

    def get_generation_config(self) -> GenerationConfig:
        grid = self.panels['grid'].get_wavelength_grid()
        channels = self.panels['channels'].get_channel_config()
        dimensions = self.panels['dimensions'].get_dimensions()
        seed = self.panels['seed'].get_seed()
        return GenerationConfig(seed=seed, grid=grid, channels=channels, dimensions=dimensions)

    def get_fluorophores(self) -> list[Fluorophore]:
        return self.fluor_manager.get_fluorophores()

    def get_object_specs(self) -> list[dict]:
        return self.object_manager.get_objects()

    def apply_dataset(self, spectral_system, field_spec) -> None:
        try:
            lambdas = spectral_system.lambdas
            if lambdas.size > 1:
                step = float(np.median(np.diff(lambdas)))
                self.panels['grid'].grid_start.set(float(lambdas[0]))
                self.panels['grid'].grid_stop.set(float(lambdas[-1]))
                self.panels['grid'].grid_step.set(step)
            
            # Update channels with loaded data
            channels = spectral_system.channels
            if channels:
                channel_configs = [
                    {
                        'name': ch.name,
                        'center_nm': ch.center_nm,
                        'bandwidth_nm': ch.bandwidth_nm
                    }
                    for ch in channels
                ]
                self.panels['channels'].manager.set_channels(channel_configs)
            
            H, W = field_spec.shape
            dims_panel = self.panels['dimensions']
            dims_panel.H.set(int(H))
            dims_panel.W.set(int(W))
            dims_panel.pixel_nm.set(float(field_spec.pixel_size_nm))
        except Exception:
            pass

    def set_fluorophores_from_dataset(self, fluorophores) -> None:
        self.fluor_manager.set_fluorophores(list(fluorophores))

    def _toggle_visibility(self) -> None:
        if self._visible:
            self._settings_container.grid_remove()
            self._toggle_button.config(text='Show Settings')
        else:
            self._settings_container.grid()
            self._toggle_button.config(text='Hide Settings')
        self._visible = not self._visible
