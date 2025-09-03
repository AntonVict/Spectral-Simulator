"""Main GUI for the spectral unmixing playground."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from spectral_playground.core.spectra import SpectralSystem, Channel, Fluorophore
from spectral_playground.core.spatial import FieldSpec, AbundanceField
from spectral_playground.core.background import BackgroundModel
from spectral_playground.core.noise import NoiseModel
from spectral_playground.core.simulate import ForwardConfig, ForwardModel
from spectral_playground.eval.metrics import rmse, sam
from spectral_playground.experiments.registry import make_unmixer

from .fluorophore_editor import FluorophoreListManager
from .object_layers import ObjectLayersManager
from .settings_panels import (
    WavelengthGridPanel, DetectionChannelsPanel, SpatialFieldPanel,
    NoiseModelPanel, UnmixingMethodsPanel, RandomSeedPanel
)


class PlaygroundGUI(tk.Tk):
    """Main GUI application for the spectral unmixing playground."""
    
    def __init__(self) -> None:
        super().__init__()
        self.title("Spectral Unmixing Playground")
        self.geometry("1400x900")
        
        # State
        self.current_Y = None
        self.current_A = None
        self.current_B = None
        self.current_M = None
        self.current_field = None
        self.current_spectral = None
        self.channel_vars = []
        self.fluor_vars = []
        self.settings_visible = True
        self.show_regions = tk.BooleanVar(value=False)
        
        # Initialize components
        self.settings_panels = {}
        self.fluor_manager = None
        self.object_manager = None
        
        self._build_widgets()

    def _build_widgets(self) -> None:
        """Build the main widget layout."""
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        self._build_left_panel()
        self._build_right_panel()
        self._build_bottom_panel()

    def _build_left_panel(self):
        """Build the left settings panel."""
        # Left panel with scrollable settings
        left_frame = ttk.Frame(self)
        left_frame.grid(row=0, column=0, sticky="nsw", padx=4, pady=4)
        
        # Toggle settings button (always visible)
        toggle_frame = ttk.Frame(left_frame)
        toggle_frame.pack(fill=tk.X, pady=(0, 4))
        self.toggle_btn = ttk.Button(toggle_frame, text="Hide Settings", command=self._toggle_settings)
        self.toggle_btn.pack(side=tk.LEFT)
        
        # Scrollable settings container with more width
        self.settings_container = ttk.Frame(left_frame)
        self.settings_container.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(self.settings_container, width=350)  # Increased width
        scrollbar = ttk.Scrollbar(self.settings_container, orient="vertical", command=canvas.yview)
        self.settings_frame = ttk.Frame(canvas)
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)
        
        self.settings_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.settings_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self._build_settings()

    def _build_right_panel(self):
        """Build the right panel with data visualization."""
        # Right notebook with two tabs: Data and Unmix
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        self.tab_data = ttk.Frame(self.notebook)
        self.tab_unmix = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_data, text="1) Data")
        self.notebook.add(self.tab_unmix, text="2) Unmix")
        
        # Data tab with visualization controls at top
        data_controls = ttk.Frame(self.tab_data)
        data_controls.pack(fill=tk.X, padx=8, pady=4)
        
        # Channel controls
        ch_control_frame = ttk.LabelFrame(data_controls, text="Channel Visibility")
        ch_control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,8))
        
        self.channel_checks_frame = ttk.Frame(ch_control_frame)
        self.channel_checks_frame.pack(padx=4, pady=2)
        
        ch_btn_frame = ttk.Frame(ch_control_frame)
        ch_btn_frame.pack(padx=4, pady=2)
        ttk.Button(ch_btn_frame, text="All", command=self._select_all_channels, width=6).pack(side=tk.LEFT, padx=(0,2))
        ttk.Button(ch_btn_frame, text="None", command=self._select_no_channels, width=6).pack(side=tk.LEFT)
        
        # Fluorophore controls
        fl_control_frame = ttk.LabelFrame(data_controls, text="Fluorophore Visibility")
        fl_control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,8))
        
        self.fluor_checks_frame = ttk.Frame(fl_control_frame)
        self.fluor_checks_frame.pack(padx=4, pady=2)
        
        fl_btn_frame = ttk.Frame(fl_control_frame)
        fl_btn_frame.pack(padx=4, pady=2)
        ttk.Button(fl_btn_frame, text="All", command=self._select_all_fluors, width=6).pack(side=tk.LEFT, padx=(0,2))
        ttk.Button(fl_btn_frame, text="None", command=self._select_no_fluors, width=6).pack(side=tk.LEFT)

        # Show regions overlay toggle
        ttk.Checkbutton(data_controls, text="Show Regions", variable=self.show_regions, command=self._render_data_view).pack(side=tk.LEFT, padx=(8,0))
        
        # Data tab figure
        self.data_figure = Figure(figsize=(10, 8), dpi=100)
        self.data_canvas = FigureCanvasTkAgg(self.data_figure, master=self.tab_data)
        self.data_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Unmix tab figure and output
        self.unmix_figure = Figure(figsize=(10, 6), dpi=100)
        self.unmix_canvas = FigureCanvasTkAgg(self.unmix_figure, master=self.tab_unmix)
        self.unmix_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_bottom_panel(self):
        """Build the bottom panel with actions and output."""
        # Bottom panel for Actions and Output
        bottom_panel = ttk.Frame(self)
        bottom_panel.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=8, pady=8)
        bottom_panel.columnconfigure(1, weight=1)
        
        # Actions panel (left side of bottom)
        actions_frame = ttk.LabelFrame(bottom_panel, text="Actions")
        actions_frame.grid(row=0, column=0, sticky="nsew", padx=(0,8))
        
        action_btn_frame = ttk.Frame(actions_frame)
        action_btn_frame.pack(padx=8, pady=8)
        ttk.Button(action_btn_frame, text="Generate Data", command=self.on_generate, width=15).pack(pady=(0,4))
        ttk.Button(action_btn_frame, text="Run Unmix", command=self.on_unmix, width=15).pack()
        
        # Output terminal (right side of bottom)
        output_frame = ttk.LabelFrame(bottom_panel, text="Output")
        output_frame.grid(row=0, column=1, sticky="nsew")
        
        self.output = tk.Text(output_frame, height=6)
        output_scroll = ttk.Scrollbar(output_frame, orient="vertical", command=self.output.yview)
        self.output.configure(yscrollcommand=output_scroll.set)
        
        self.output.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        output_scroll.pack(side="right", fill="y", pady=4)

    def _build_settings(self) -> None:
        """Build all settings panels."""
        row = 0
        
        # Wavelength Grid
        grid_group = ttk.LabelFrame(self.settings_frame, text="Wavelength Grid")
        grid_group.grid(row=row, column=0, sticky="ew", padx=2, pady=1)
        grid_group.columnconfigure(0, weight=1)
        self.settings_panels['grid'] = WavelengthGridPanel(grid_group)
        row += 1
        
        # Detection Channels
        ch_group = ttk.LabelFrame(self.settings_frame, text="Detection Channels")
        ch_group.grid(row=row, column=0, sticky="ew", padx=2, pady=1)
        ch_group.columnconfigure(0, weight=1)
        self.settings_panels['channels'] = DetectionChannelsPanel(ch_group)
        row += 1
        
        # Fluorophores
        fluor_group = ttk.LabelFrame(self.settings_frame, text="Fluorophores")
        fluor_group.grid(row=row, column=0, sticky="ew", padx=2, pady=1)
        fluor_group.columnconfigure(0, weight=1)
        fluor_main_frame = ttk.Frame(fluor_group)
        fluor_main_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.fluor_manager = FluorophoreListManager(fluor_main_frame, self._log)
        row += 1
        
        # Spatial Field
        spatial_group = ttk.LabelFrame(self.settings_frame, text="Spatial Field")
        spatial_group.grid(row=row, column=0, sticky="ew", padx=2, pady=1)
        spatial_group.columnconfigure(0, weight=1)
        self.settings_panels['spatial'] = SpatialFieldPanel(spatial_group)
        row += 1
        
        # Object Layers
        objects_group = ttk.LabelFrame(self.settings_frame, text="Object Layers")
        objects_group.grid(row=row, column=0, sticky="ew", padx=2, pady=1)
        objects_group.columnconfigure(0, weight=1)
        self.object_manager = ObjectLayersManager(objects_group, self._log, self._get_image_dims)
        row += 1
        
        # Noise Model
        noise_group = ttk.LabelFrame(self.settings_frame, text="Noise Model")
        noise_group.grid(row=row, column=0, sticky="ew", padx=2, pady=1)
        noise_group.columnconfigure(0, weight=1)
        self.settings_panels['noise'] = NoiseModelPanel(noise_group, None)
        row += 1
        
        # Unmixing Methods
        methods_group = ttk.LabelFrame(self.settings_frame, text="Unmixing Methods")
        methods_group.grid(row=row, column=0, sticky="ew", padx=2, pady=1)
        methods_group.columnconfigure(0, weight=1)
        self.settings_panels['methods'] = UnmixingMethodsPanel(methods_group)
        row += 1
        
        # Random Seed
        seed_group = ttk.LabelFrame(self.settings_frame, text="Random Seed")
        seed_group.grid(row=row, column=0, sticky="ew", padx=2, pady=1)
        seed_group.columnconfigure(0, weight=1)
        self.settings_panels['seed'] = RandomSeedPanel(seed_group)
        row += 1

    def _get_image_dims(self):
        """Get current image dimensions."""
        spatial_config = self.settings_panels['spatial'].get_spatial_config()
        dims = spatial_config['dimensions']
        return dims['H'], dims['W']

    def _log(self, msg: str) -> None:
        """Log a message to the output terminal."""
        self.output.insert(tk.END, msg + "\n")
        self.output.see(tk.END)

    def _toggle_settings(self) -> None:
        """Toggle settings panel visibility."""
        if self.settings_visible:
            self.settings_container.pack_forget()
            self.toggle_btn.config(text="Show Settings")
            self.settings_visible = False
        else:
            self.settings_container.pack(fill=tk.BOTH, expand=True)
            self.toggle_btn.config(text="Hide Settings")
            self.settings_visible = True

    def on_generate(self) -> None:
        """Generate synthetic data based on current settings."""
        try:
            # Get configurations from all panels
            seed = self.settings_panels['seed'].get_seed()
            rng = np.random.default_rng(seed)
            
            grid_config = self.settings_panels['grid'].get_wavelength_grid()
            ch_config = self.settings_panels['channels'].get_channel_config()
            spatial_config = self.settings_panels['spatial'].get_spatial_config()
            noise_config = self.settings_panels['noise'].get_noise_config()
            
            # Build wavelength grid
            lambdas = np.arange(
                grid_config['start'], 
                grid_config['stop'] + 1e-9, 
                grid_config['step'], 
                dtype=np.float32
            )

            # Build channels
            L = ch_config['count']
            bw = ch_config['bandwidth']
            centers = np.linspace(
                grid_config['start'] + 0.5 * bw, 
                grid_config['stop'] - 0.5 * bw, 
                L
            )
            channels = [Channel(name=f"C{idx+1}", center_nm=float(c), bandwidth_nm=bw) 
                       for idx, c in enumerate(centers)]

            # Get fluorophores
            fluors = self.fluor_manager.get_fluorophores()
            K = len(fluors)

            spectral = SpectralSystem(lambdas=lambdas, channels=channels, fluors=fluors)
            M = spectral.build_M()

            # Build field
            dims = spatial_config['dimensions']
            H, W = dims['H'], dims['W']
            px_nm = dims['pixel_nm']
            field = FieldSpec(shape=(H, W), pixel_size_nm=px_nm)
            
            # Generate abundance field
            af = AbundanceField(rng)
            
            # Build base/global field if enabled
            base_A = None
            if self.object_manager.should_include_base_field():
                global_field = spatial_config['global_field']
                kind = global_field['kind']
                
                if kind == "dots":
                    base_A = af.sample(
                        K=K, field=field, kind="dots",
                        density_per_100x100_um2=global_field['density'],
                        spot_profile={"kind": "gaussian", "sigma_px": global_field['spot_sigma']},
                    )
                elif kind == "uniform":
                    base_A = af.sample(K=K, field=field, kind="uniform")
                else:
                    base_A = af.sample(
                        K=K, field=field, kind=kind,
                        count_per_fluor=global_field['count_per_fluor'],
                        size_px=global_field['size_px'],
                        intensity_min=global_field['intensity_min'],
                        intensity_max=global_field['intensity_max'],
                    )

            # If objects exist, build from objects (with optional base)
            objects = self.object_manager.get_objects()
            if len(objects) > 0:
                A = af.build_from_objects(K=K, field=field, objects=objects, base=base_A)
            else:
                A = base_A if base_A is not None else af.sample(K=K, field=field, kind="uniform")

            # Generate background and noise
            bg = BackgroundModel(rng)
            noise = NoiseModel(rng)
            fwd = ForwardModel(spectral=spectral, field=field, bg=bg, noise=noise, cfg=ForwardConfig())
            B = bg.sample(M.shape[0], H, W, kind="constant", level=0.0)
            
            # Apply noise based on toggle
            if noise_config['enabled']:
                noise_params = {
                    "gain": noise_config['gain'], 
                    "read_sigma": noise_config['read_sigma'], 
                    "dark_rate": noise_config['dark_rate']
                }
            else:
                # No noise - perfect signal
                noise_params = {"gain": 1.0, "read_sigma": 0.0, "dark_rate": 0.0}
            
            Y = fwd.synthesize(A, B=B, noise_params=noise_params)

            # Save state
            self.current_Y = Y
            self.current_A = A
            self.current_B = B
            self.current_M = M
            self.current_field = field
            self.current_spectral = spectral

            # Build toggles and render data view
            self._populate_channel_toggles(L)
            self._populate_fluor_toggles(K)
            self._render_data_view()

            noise_status = "enabled" if noise_config['enabled'] else "disabled"
            self._log(f"Generated data: L={L}, K={K}, HxW={H}x{W}, noise={noise_status}")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self._log(f"Error generating data: {str(e)}")

    def on_unmix(self) -> None:
        """Run unmixing algorithms on the current data."""
        try:
            if self.current_Y is None:
                messagebox.showinfo("Info", "Generate data first.")
                return
                
            Y = self.current_Y
            M = self.current_M
            A = self.current_A
            B = self.current_B
            H, W = self.current_field.shape

            methods = self.settings_panels['methods'].get_methods_config()

            self.output.delete("1.0", tk.END)
            self._log(f"Unmixing with {len(methods)} method(s)")
            self.unmix_figure.clear()
            
            if len(methods) == 0:
                self._log("No methods selected!")
                return
                
            n_methods = len(methods)
            for i, spec in enumerate(methods):
                un = make_unmixer(spec)
                known_M = None if (spec.get("method") == "nmf" and spec.get("K") is not None) else M
                un.fit(Y, M=known_M)
                out = un.transform(Y, M=known_M)
                A_hat = out.get("A")
                M_hat = out.get("M", known_M)
                m_rmse = rmse(Y, (M_hat @ A_hat) + B) if (A_hat is not None and M_hat is not None) else None
                a_rmse = rmse(A_hat, A) if A_hat is not None else None
                m_sam = sam(M_hat, M, axis=0) if M_hat is not None else None
                self._log(f"{un.name}: rmse_Y={m_rmse:.4f}  rmse_A={a_rmse:.4f}  sam_M={m_sam:.4f}")

                if A_hat is not None:
                    K_show = min(A_hat.shape[0], 3)
                    for k in range(K_show):
                        ax = self.unmix_figure.add_subplot(K_show, n_methods, k * n_methods + i + 1)
                        ax.imshow(A_hat[k].reshape(H, W), cmap="magma")
                        ax.set_title(f"{un.name}: A[{k}]", fontsize=10)
                        ax.axis("off")
            
            self.unmix_canvas.draw_idle()

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self._log(f"Error during unmixing: {str(e)}")

    def _populate_channel_toggles(self, L: int) -> None:
        """Populate channel visibility toggles."""
        for w in self.channel_checks_frame.winfo_children():
            w.destroy()
        self.channel_vars = []
        for i in range(L):
            var = tk.BooleanVar(value=True)
            self.channel_vars.append(var)
            ttk.Checkbutton(self.channel_checks_frame, text=f"C{i+1}", variable=var, command=self._render_data_view).grid(row=i//4, column=i%4, sticky="w")

    def _select_all_channels(self) -> None:
        """Select all channels."""
        for v in self.channel_vars:
            v.set(True)
        self._render_data_view()

    def _select_no_channels(self) -> None:
        """Deselect all channels."""
        for v in self.channel_vars:
            v.set(False)
        self._render_data_view()

    def _populate_fluor_toggles(self, K: int) -> None:
        """Populate fluorophore visibility toggles."""
        for w in self.fluor_checks_frame.winfo_children():
            w.destroy()
        self.fluor_vars = []
        for i in range(K):
            var = tk.BooleanVar(value=True)
            self.fluor_vars.append(var)
            ttk.Checkbutton(self.fluor_checks_frame, text=f"F{i+1}", variable=var, command=self._render_data_view).grid(row=i//4, column=i%4, sticky="w")

    def _select_all_fluors(self) -> None:
        """Select all fluorophores."""
        for v in self.fluor_vars:
            v.set(True)
        self._render_data_view()

    def _select_no_fluors(self) -> None:
        """Deselect all fluorophores."""
        for v in self.fluor_vars:
            v.set(False)
        self._render_data_view()

    def _render_data_view(self) -> None:
        """Render the data visualization."""
        if self.current_Y is None or self.current_field is None or self.current_spectral is None:
            return
        H, W = self.current_field.shape
        L, P = self.current_Y.shape
        active_ch = [i for i, v in enumerate(self.channel_vars) if v.get()]
        active_fl = [i for i, v in enumerate(self.fluor_vars) if v.get()]
        
        # Build RGB composite
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        eps = 1e-6
        for i in active_ch:
            img = self.current_Y[i].reshape(H, W)
            img = img / (np.percentile(img, 99.0) + eps)
            img = np.clip(img, 0.0, 1.0)
            nm = self.current_spectral.channels[i].center_nm
            color = np.array(self._wavelength_to_rgb_nm(nm), dtype=np.float32)
            rgb += img[..., None] * color[None, None, :]
        rgb = np.clip(rgb, 0.0, 1.0)

        # Clear and create improved layout
        self.data_figure.clear()
        
        # Main composite image (larger)
        ax_img = self.data_figure.add_subplot(2, 2, (1, 2))
        ax_img.imshow(rgb)
        ax_img.set_title("Composite Image (Selected Channels)", fontsize=12)
        ax_img.axis("off")

        # Optional region overlay for objects
        if self.show_regions.get() and len(self.object_manager.get_objects()) > 0:
            overlay = np.zeros((H, W, 3), dtype=np.float32)
            # Cycle colors
            colors = [np.array([1.0,0.0,0.0]), np.array([0.0,1.0,0.0]), np.array([0.0,0.0,1.0]), np.array([1.0,1.0,0.0])]
            color_idx = 0
            for obj in self.object_manager.get_objects():
                col = colors[color_idx % len(colors)]
                color_idx += 1
                region = obj.get('region', {'type': 'full'})
                rtype = region.get('type','full')
                if rtype == 'rect':
                    x0 = int(max(0, region.get('x0', 0)))
                    y0 = int(max(0, region.get('y0', 0)))
                    w = int(max(1, region.get('w', W)))
                    h = int(max(1, region.get('h', H)))
                    x1 = min(W, x0 + w)
                    y1 = min(H, y0 + h)
                    overlay[y0:y1, x0:x1, :] += col * 0.15
                elif rtype == 'circle':
                    cx = float(region.get('cx', W/2))
                    cy = float(region.get('cy', H/2))
                    r = float(region.get('r', min(H,W)/3))
                    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
                    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (r ** 2)
                    overlay[mask] += col * 0.15
                elif rtype == 'full':
                    overlay[:, :, :] += col * 0.05
            overlay = np.clip(overlay, 0.0, 1.0)
            ax_img.imshow(np.clip(rgb + overlay, 0.0, 1.0))

        # Spectral panel
        self._render_spectral_panel(active_ch, active_fl)
        
        # Abundance maps
        self._render_abundance_maps(active_fl, H, W)
        
        self.data_canvas.draw_idle()

    def _render_spectral_panel(self, active_ch, active_fl):
        """Render the spectral profiles panel."""
        ax_spec = self.data_figure.add_subplot(2, 2, 3)
        grid_config = self.settings_panels['grid'].get_wavelength_grid()
        start = grid_config['start']
        stop = grid_config['stop']
        ax_spec.set_xlim(start, stop)
        ax_spec.set_ylim(0, 1.1)
        ax_spec.set_xlabel("Wavelength (nm)")
        ax_spec.set_ylabel("Normalized Intensity")
        ax_spec.set_title("Spectral Profiles", fontsize=10)
        
        # Draw channel bands
        for idx, ch in enumerate(self.current_spectral.channels):
            half = 0.5 * ch.bandwidth_nm
            low, high = ch.center_nm - half, ch.center_nm + half
            active_flag = idx in active_ch
            if active_flag:
                face = self._wavelength_to_rgb_nm(ch.center_nm)
                ax_spec.axvspan(low, high, ymin=0, ymax=0.15, facecolor=face, alpha=0.7, edgecolor='k', linewidth=1)
                ax_spec.text(ch.center_nm, 0.08, f"C{idx+1}", ha='center', va='center', fontsize=8, weight='bold')
            else:
                ax_spec.axvspan(low, high, ymin=0, ymax=0.15, facecolor='lightgray', alpha=0.3, edgecolor='gray')
                
        # Overlay fluorophore PDFs and measured total signal
        if self.current_A is not None and self.current_M is not None:
            K = self.current_A.shape[0]
            
            # Show individual fluorophore spectral profiles (theoretical)
            for k in range(K):
                show = k in active_fl
                if show:
                    pdf = self.current_spectral._pdf(self.current_spectral.fluors[k])
                    # Normalize PDF for display
                    pdf_norm = pdf / (np.max(pdf) + 1e-9)
                    peak_nm = float(self.current_spectral.lambdas[np.argmax(pdf)])
                    col = self._wavelength_to_rgb_nm(peak_nm)
                    ax_spec.plot(self.current_spectral.lambdas, 0.2 + 0.6 * pdf_norm, 
                               color=col, linewidth=2, label=f"F{k+1} (theory)", alpha=0.8)
            
            # Calculate and show actual measured total signal as a continuous curve
            total_per_channel = np.sum(self.current_Y, axis=1)  # Shape: (L,)
            channel_centers = np.array([ch.center_nm for ch in self.current_spectral.channels])
            
            # Interpolate the discrete channel totals across the wavelength grid
            if len(channel_centers) >= 2:
                interp = np.interp(self.current_spectral.lambdas, channel_centers, total_per_channel, 
                                   left=total_per_channel[0], right=total_per_channel[-1])
            else:
                interp = np.full_like(self.current_spectral.lambdas, total_per_channel[0] if len(total_per_channel) > 0 else 0.0)

            # Simple smoothing via moving average over ~5 nm window
            lambdas = self.current_spectral.lambdas
            if len(lambdas) > 3:
                step_nm = max(1.0, float(np.median(np.diff(lambdas))))
                window = max(3, int(round(5.0 / step_nm)) | 1)  # odd
                kernel = np.ones(window, dtype=np.float32)
                kernel /= np.sum(kernel)
                interp_smooth = np.convolve(interp, kernel, mode='same')
            else:
                interp_smooth = interp

            # Normalize for display and plot
            if np.max(interp_smooth) > 0:
                curve = 0.2 + 0.6 * (interp_smooth / np.max(interp_smooth))
                ax_spec.plot(self.current_spectral.lambdas, curve, color='black', linewidth=2.5, label='Measured Total (cont.)', alpha=0.9)
            else:
                ax_spec.plot(self.current_spectral.lambdas, np.zeros_like(self.current_spectral.lambdas) + 0.2, color='black', linewidth=2.5, label='Measured Total (cont.)', alpha=0.9)
                           
            ax_spec.legend(loc='upper right', fontsize=8, framealpha=0.8)

    def _render_abundance_maps(self, active_fl, H, W):
        """Render abundance maps for selected fluorophores."""
        if self.current_A is not None and len(active_fl) > 0:
            ax_maps = self.data_figure.add_subplot(2, 2, 4)
            n_show = min(len(active_fl), 1)  # Show only one at a time for clarity
            if n_show > 0:
                k = active_fl[0]  # Show first selected
                im = ax_maps.imshow(self.current_A[k].reshape(H, W), cmap="magma")
                ax_maps.set_title(f"Ground Truth: F{k+1} Abundance", fontsize=10)
                ax_maps.axis("off")
                self.data_figure.colorbar(im, ax=ax_maps, fraction=0.046, pad=0.04)

    @staticmethod
    def _wavelength_to_rgb_nm(nm: float) -> tuple[float, float, float]:
        """Convert wavelength to RGB color."""
        # Approximate wavelength to RGB (380-780nm)
        w = nm
        if w < 380 or w > 780:
            return (0.5, 0.5, 0.5)
        if w < 440:
            r = -(w - 440) / (440 - 380)
            g = 0.0
            b = 1.0
        elif w < 490:
            r = 0.0
            g = (w - 440) / (490 - 440)
            b = 1.0
        elif w < 510:
            r = 0.0
            g = 1.0
            b = -(w - 510) / (510 - 490)
        elif w < 580:
            r = (w - 510) / (580 - 510)
            g = 1.0
            b = 0.0
        elif w < 645:
            r = 1.0
            g = -(w - 645) / (645 - 580)
            b = 0.0
        else:
            r = 1.0
            g = 0.0
            b = 0.0
        # Intensity factor near spectrum edges
        if w < 420:
            f = 0.3 + 0.7 * (w - 380) / (420 - 380)
        elif w > 700:
            f = 0.3 + 0.7 * (780 - w) / (780 - 700)
        else:
            f = 1.0
        return (float(r * f), float(g * f), float(b * f))


def main() -> None:
    """Main entry point for the GUI application."""
    app = PlaygroundGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
