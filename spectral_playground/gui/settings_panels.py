"""Settings panel components for the spectral visualization GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class WavelengthGridPanel:
    """Panel for wavelength grid settings."""

    def __init__(self, parent_frame):
        self.grid_start = tk.DoubleVar(value=300.0)
        self.grid_stop = tk.DoubleVar(value=850.0)
        self.grid_step = tk.DoubleVar(value=1.0)
        self._build_ui(parent_frame)

    def _build_ui(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=2, pady=2)
        ttk.Label(frame, text='Start').grid(row=0, column=0, sticky='w')
        ttk.Label(frame, text='Stop').grid(row=0, column=1, sticky='w')
        ttk.Label(frame, text='Step').grid(row=0, column=2, sticky='w')
        ttk.Entry(frame, textvariable=self.grid_start, width=8).grid(row=1, column=0, padx=(0, 2))
        ttk.Entry(frame, textvariable=self.grid_stop, width=8).grid(row=1, column=1, padx=(0, 2))
        ttk.Entry(frame, textvariable=self.grid_step, width=8).grid(row=1, column=2)

    def get_wavelength_grid(self):
        return {
            'start': float(self.grid_start.get()),
            'stop': float(self.grid_stop.get()),
            'step': float(self.grid_step.get()),
        }


class DetectionChannelsPanel:
    """Panel for detection channel settings."""

    def __init__(self, parent_frame, log_callback, get_wavelength_range_callback):
        from .channel_manager import ChannelListManager
        self.manager = ChannelListManager(
            parent_frame,
            log_callback,
            get_wavelength_range_callback
        )

    def get_channel_config(self):
        """Get list of channel configurations.
        
        Returns:
            List of channel configuration dictionaries
        """
        return self.manager.get_all_channels()


class RandomSeedPanel:
    """Panel for random seed settings."""

    def __init__(self, parent_frame):
        self.seed = tk.IntVar(value=123)
        self._build_ui(parent_frame)

    def _build_ui(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=2, pady=2)
        ttk.Label(frame, text='Seed').grid(row=0, column=0, sticky='w')
        ttk.Entry(frame, textvariable=self.seed, width=12).grid(row=1, column=0, pady=(2, 0))

    def get_seed(self):
        return int(self.seed.get())


class ImageDimensionsPanel:
    """Panel for basic image dimensions settings."""

    def __init__(self, parent_frame):
        self.H = tk.IntVar(value=128)
        self.W = tk.IntVar(value=128)
        self.pixel_nm = tk.DoubleVar(value=100.0)
        self._build_ui(parent_frame)

    def _build_ui(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=2, pady=2)
        ttk.Label(frame, text='H').grid(row=0, column=0, sticky='w')
        ttk.Label(frame, text='W').grid(row=0, column=1, sticky='w')
        ttk.Label(frame, text='Pixel (nm)').grid(row=0, column=2, sticky='w')
        ttk.Spinbox(frame, textvariable=self.H, from_=32, to=1024, width=6, increment=32).grid(row=1, column=0, padx=(0, 2))
        ttk.Spinbox(frame, textvariable=self.W, from_=32, to=1024, width=6, increment=32).grid(row=1, column=1, padx=(0, 2))
        ttk.Entry(frame, textvariable=self.pixel_nm, width=8).grid(row=1, column=2)

    def get_dimensions(self):
        return {
            'H': int(self.H.get()),
            'W': int(self.W.get()),
            'pixel_nm': float(self.pixel_nm.get()),
        }
