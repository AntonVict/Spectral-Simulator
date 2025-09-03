"""Settings panel components for the spectral unmixing playground GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class WavelengthGridPanel:
    """Panel for wavelength grid settings."""
    
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.grid_start = tk.DoubleVar(value=450.0)
        self.grid_stop = tk.DoubleVar(value=700.0)
        self.grid_step = tk.DoubleVar(value=1.0)
        self._build_ui()
        
    def _build_ui(self):
        """Build the wavelength grid UI."""
        grid_frame = ttk.Frame(self.parent_frame)
        grid_frame.pack(fill=tk.X, padx=2, pady=2)
        
        ttk.Label(grid_frame, text="Start").grid(row=0, column=0, sticky="w")
        ttk.Label(grid_frame, text="Stop").grid(row=0, column=1, sticky="w")
        ttk.Label(grid_frame, text="Step").grid(row=0, column=2, sticky="w")
        
        ttk.Entry(grid_frame, textvariable=self.grid_start, width=8).grid(row=1, column=0, padx=(0,2))
        ttk.Entry(grid_frame, textvariable=self.grid_stop, width=8).grid(row=1, column=1, padx=(0,2))
        ttk.Entry(grid_frame, textvariable=self.grid_step, width=8).grid(row=1, column=2)
        
    def get_wavelength_grid(self):
        """Get wavelength grid parameters."""
        return {
            'start': float(self.grid_start.get()),
            'stop': float(self.grid_stop.get()),
            'step': float(self.grid_step.get())
        }


class DetectionChannelsPanel:
    """Panel for detection channel settings."""
    
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.num_channels = tk.IntVar(value=4)
        self.bandwidth = tk.DoubleVar(value=30.0)
        self._build_ui()
        
    def _build_ui(self):
        """Build the detection channels UI."""
        ch_frame = ttk.Frame(self.parent_frame)
        ch_frame.pack(fill=tk.X, padx=2, pady=2)
        
        ttk.Label(ch_frame, text="Count (L)").grid(row=0, column=0, sticky="w")
        ttk.Label(ch_frame, text="Bandwidth (nm)").grid(row=0, column=1, sticky="w")
        
        ttk.Entry(ch_frame, textvariable=self.num_channels, width=8).grid(row=1, column=0, padx=(0,2))
        ttk.Entry(ch_frame, textvariable=self.bandwidth, width=12).grid(row=1, column=1)
        
    def get_channel_config(self):
        """Get channel configuration."""
        return {
            'count': int(self.num_channels.get()),
            'bandwidth': float(self.bandwidth.get())
        }


class SpatialFieldPanel:
    """Panel for spatial field settings."""
    
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        
        # Basic dimensions
        self.H = tk.IntVar(value=128)
        self.W = tk.IntVar(value=128)
        self.pixel_nm = tk.DoubleVar(value=100.0)
        
        # Global field settings
        self.spatial_kind = tk.StringVar(value="dots")
        self.density = tk.DoubleVar(value=50.0)
        self.spot_sigma = tk.DoubleVar(value=1.2)
        self.count_per_fluor = tk.IntVar(value=50)
        self.size_px = tk.DoubleVar(value=6.0)
        self.intensity_min = tk.DoubleVar(value=0.5)
        self.intensity_max = tk.DoubleVar(value=1.5)
        
        self._build_ui()
        
    def _build_ui(self):
        """Build the spatial field UI."""
        # Main spatial container using grid throughout
        spatial_main = ttk.Frame(self.parent_frame)
        spatial_main.pack(fill=tk.X, padx=2, pady=2)
        
        # Basic dimensions frame
        dims_frame = ttk.Frame(spatial_main)
        dims_frame.pack(fill=tk.X, pady=(0,4))
        
        ttk.Label(dims_frame, text="H").grid(row=0, column=0, sticky="w")
        ttk.Label(dims_frame, text="W").grid(row=0, column=1, sticky="w")
        ttk.Label(dims_frame, text="Pixel (nm)").grid(row=0, column=2, sticky="w")
        
        ttk.Entry(dims_frame, textvariable=self.H, width=6).grid(row=1, column=0, padx=(0,2))
        ttk.Entry(dims_frame, textvariable=self.W, width=6).grid(row=1, column=1, padx=(0,2))
        ttk.Entry(dims_frame, textvariable=self.pixel_nm, width=8).grid(row=1, column=2)
        
        # Global field parameters (compact layout)
        global_frame = ttk.LabelFrame(spatial_main, text="Global Field Settings")
        global_frame.pack(fill=tk.X, pady=(4,0))
        global_frame.columnconfigure((0,1,2), weight=1)
        
        # Row 0: Type, Density, Spot σ
        ttk.Label(global_frame, text="Type:").grid(row=0, column=0, sticky="w", padx=2)
        kind_combo = ttk.Combobox(global_frame, textvariable=self.spatial_kind,
                                  values=["dots", "uniform", "circles", "boxes", "gaussian_blobs", "mixed"],
                                  state="readonly", width=10)
        kind_combo.grid(row=1, column=0, sticky="ew", padx=2)
        
        ttk.Label(global_frame, text="Density (/100×100μm²):").grid(row=0, column=1, sticky="w", padx=2)
        ttk.Entry(global_frame, textvariable=self.density, width=8).grid(row=1, column=1, sticky="ew", padx=2)
        
        ttk.Label(global_frame, text="Spot σ (px):").grid(row=0, column=2, sticky="w", padx=2)
        ttk.Entry(global_frame, textvariable=self.spot_sigma, width=8).grid(row=1, column=2, sticky="ew", padx=2)

        # Row 1: Count, Size, Intensity
        ttk.Label(global_frame, text="Count/fluor:").grid(row=2, column=0, sticky="w", padx=2, pady=(4,0))
        ttk.Entry(global_frame, textvariable=self.count_per_fluor, width=8).grid(row=3, column=0, sticky="ew", padx=2)

        ttk.Label(global_frame, text="Size (px):").grid(row=2, column=1, sticky="w", padx=2, pady=(4,0))
        ttk.Entry(global_frame, textvariable=self.size_px, width=8).grid(row=3, column=1, sticky="ew", padx=2)

        # Intensity range in one column
        ttk.Label(global_frame, text="Intensity min-max:").grid(row=2, column=2, sticky="w", padx=2, pady=(4,0))
        intensity_frame = ttk.Frame(global_frame)
        intensity_frame.grid(row=3, column=2, sticky="ew", padx=2)
        ttk.Entry(intensity_frame, textvariable=self.intensity_min, width=4).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(intensity_frame, text="-").pack(side=tk.LEFT)
        ttk.Entry(intensity_frame, textvariable=self.intensity_max, width=4).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
    def get_spatial_config(self):
        """Get spatial field configuration."""
        return {
            'dimensions': {
                'H': int(self.H.get()),
                'W': int(self.W.get()),
                'pixel_nm': float(self.pixel_nm.get())
            },
            'global_field': {
                'kind': self.spatial_kind.get(),
                'density': float(self.density.get()),
                'spot_sigma': float(self.spot_sigma.get()),
                'count_per_fluor': int(self.count_per_fluor.get()),
                'size_px': float(self.size_px.get()),
                'intensity_min': float(self.intensity_min.get()),
                'intensity_max': float(self.intensity_max.get())
            }
        }


class NoiseModelPanel:
    """Panel for noise model settings."""
    
    def __init__(self, parent_frame, on_noise_toggle_callback):
        self.parent_frame = parent_frame
        self.on_noise_toggle = on_noise_toggle_callback
        
        self.noise_enabled = tk.BooleanVar(value=True)
        self.gain = tk.DoubleVar(value=1.0)
        self.read_sigma = tk.DoubleVar(value=1.0)
        self.dark_rate = tk.DoubleVar(value=0.0)
        
        self._build_ui()
        
    def _build_ui(self):
        """Build the noise model UI."""
        noise_main_frame = ttk.Frame(self.parent_frame)
        noise_main_frame.pack(fill=tk.X, padx=2, pady=2)
        
        # Noise enable/disable toggle
        noise_toggle = ttk.Checkbutton(noise_main_frame, text="Enable Noise", variable=self.noise_enabled, command=self._on_noise_toggle)
        noise_toggle.pack(anchor='w', pady=(0, 4))
        
        # Noise parameters frame
        self.noise_params_frame = ttk.Frame(noise_main_frame)
        self.noise_params_frame.pack(fill=tk.X)
        
        ttk.Label(self.noise_params_frame, text="Gain").grid(row=0, column=0, sticky="w")
        ttk.Label(self.noise_params_frame, text="Read σ").grid(row=0, column=1, sticky="w")
        ttk.Label(self.noise_params_frame, text="Dark rate").grid(row=0, column=2, sticky="w")
        
        self.gain_entry = ttk.Entry(self.noise_params_frame, textvariable=self.gain, width=6)
        self.read_entry = ttk.Entry(self.noise_params_frame, textvariable=self.read_sigma, width=6)
        self.dark_entry = ttk.Entry(self.noise_params_frame, textvariable=self.dark_rate, width=6)
        
        self.gain_entry.grid(row=1, column=0, padx=(0,2))
        self.read_entry.grid(row=1, column=1, padx=(0,2))
        self.dark_entry.grid(row=1, column=2)
        
    def _on_noise_toggle(self):
        """Handle noise toggle."""
        state = 'normal' if self.noise_enabled.get() else 'disabled'
        self.gain_entry.config(state=state)
        self.read_entry.config(state=state)
        self.dark_entry.config(state=state)
        
        if self.on_noise_toggle:
            self.on_noise_toggle()
            
    def get_noise_config(self):
        """Get noise configuration."""
        return {
            'enabled': self.noise_enabled.get(),
            'gain': float(self.gain.get()),
            'read_sigma': float(self.read_sigma.get()),
            'dark_rate': float(self.dark_rate.get())
        }


class UnmixingMethodsPanel:
    """Panel for unmixing methods settings."""
    
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        
        self.use_nnls = tk.BooleanVar(value=True)
        self.use_lasso = tk.BooleanVar(value=False)
        self.use_nmf = tk.BooleanVar(value=False)
        self.use_em = tk.BooleanVar(value=False)
        
        self.lasso_alpha = tk.DoubleVar(value=0.01)
        self.nmf_iters = tk.IntVar(value=200)
        self.em_iters = tk.IntVar(value=150)
        
        self._build_ui()
        
    def _build_ui(self):
        """Build the unmixing methods UI."""
        methods_frame = ttk.Frame(self.parent_frame)
        methods_frame.pack(fill=tk.X, padx=2, pady=2)
        
        ttk.Checkbutton(methods_frame, text="NNLS", variable=self.use_nnls).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(methods_frame, text="LASSO", variable=self.use_lasso).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(methods_frame, text="NMF", variable=self.use_nmf).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(methods_frame, text="EM", variable=self.use_em).grid(row=1, column=1, sticky="w")
        
        ttk.Label(methods_frame, text="LASSO α").grid(row=2, column=0, sticky="w", pady=(4,0))
        ttk.Label(methods_frame, text="NMF iters").grid(row=2, column=1, sticky="w", pady=(4,0))
        
        ttk.Entry(methods_frame, textvariable=self.lasso_alpha, width=8).grid(row=3, column=0, padx=(0,2))
        ttk.Entry(methods_frame, textvariable=self.nmf_iters, width=8).grid(row=3, column=1)
        
    def get_methods_config(self):
        """Get unmixing methods configuration."""
        methods = []
        if self.use_nnls.get():
            methods.append({"method": "nnls"})
        if self.use_lasso.get():
            methods.append({"method": "lasso", "alpha": float(self.lasso_alpha.get())})
        if self.use_nmf.get():
            methods.append({"method": "nmf", "n_iter": int(self.nmf_iters.get())})
        if self.use_em.get():
            methods.append({"method": "em_poisson", "n_iter": int(self.em_iters.get())})
        return methods


class RandomSeedPanel:
    """Panel for random seed settings."""
    
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.seed = tk.IntVar(value=123)
        self._build_ui()
        
    def _build_ui(self):
        """Build the random seed UI."""
        seed_frame = ttk.Frame(self.parent_frame)
        seed_frame.pack(fill=tk.X, padx=2, pady=2)
        
        ttk.Label(seed_frame, text="Seed").grid(row=0, column=0, sticky="w")
        ttk.Entry(seed_frame, textvariable=self.seed, width=12).grid(row=1, column=0, pady=(2,0))
        
    def get_seed(self):
        """Get random seed."""
        return int(self.seed.get())
