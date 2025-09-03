"""Main GUI for the spectral unmixing playground."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
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
    WavelengthGridPanel, DetectionChannelsPanel, ImageDimensionsPanel,
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
        self.spectral_fluor_vars = []  # For spectral profile visibility
        self.settings_visible = True
        
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
        
        # Scrollable settings container with proper layout
        self.settings_container = ttk.Frame(left_frame)
        self.settings_container.pack(fill=tk.BOTH, expand=True)
        
        # Create a frame to hold canvas and scrollbar side by side
        scroll_frame = ttk.Frame(self.settings_container)
        scroll_frame.pack(fill=tk.BOTH, expand=True)
        scroll_frame.grid_columnconfigure(0, weight=1)
        scroll_frame.grid_rowconfigure(0, weight=1)
        
        canvas = tk.Canvas(scroll_frame, width=330)  # Slightly reduced to leave room for scrollbar
        scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)
        self.settings_frame = ttk.Frame(canvas)
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)
        
        self.settings_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.settings_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Use grid to prevent overlap
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
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
        
        # Data tab main container with side-by-side layout
        data_main = ttk.Frame(self.tab_data)
        data_main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        data_main.columnconfigure(0, weight=1)  # Image gets most space
        data_main.columnconfigure(1, weight=0)  # Controls get fixed space
        
        # Main image area (left side)
        image_frame = ttk.Frame(data_main)
        image_frame.grid(row=0, column=0, sticky="nsew", padx=(0,8))
        
        # Image controls
        image_controls = ttk.Frame(image_frame)
        image_controls.pack(fill=tk.X, pady=(0,4))
        
        # Navigation help text
        help_text = ttk.Label(image_controls, text="Scroll: Zoom | Shift+Scroll: Horizontal | Ctrl+Scroll: Vertical | Arrow Keys: Navigate", 
                             foreground="gray", font=('TkDefaultFont', 8))
        help_text.pack(side=tk.LEFT)
        
        ttk.Button(image_controls, text="🔍 Expand View", command=self._expand_composite_view, width=15).pack(side=tk.RIGHT)
        
        self.data_figure = Figure(figsize=(10, 8), dpi=100)
        self.data_figure.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08)  # Reduce padding, leave room for toolbar
        self.data_canvas = FigureCanvasTkAgg(self.data_figure, master=image_frame)
        self.data_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar for zoom and pan
        toolbar_frame = ttk.Frame(image_frame)
        toolbar_frame.pack(fill=tk.X)
        self.data_toolbar = NavigationToolbar2Tk(self.data_canvas, toolbar_frame)
        self.data_toolbar.update()
        
        # Enable mouse wheel zooming and scrolling
        def on_scroll(event):
            ax = self.data_figure.gca()
            if ax and event.inaxes == ax:
                # Get current limits
                xlims = ax.get_xlim()
                ylims = ax.get_ylim()
                
                # Check for modifier keys
                if event.key == 'shift':
                    # Shift + scroll = horizontal scrolling
                    x_range = xlims[1] - xlims[0]
                    scroll_amount = x_range * 0.1  # 10% of current view
                    
                    if event.button == 'up':
                        # Scroll left
                        ax.set_xlim([xlims[0] - scroll_amount, xlims[1] - scroll_amount])
                    else:
                        # Scroll right
                        ax.set_xlim([xlims[0] + scroll_amount, xlims[1] + scroll_amount])
                        
                elif event.key == 'control':
                    # Ctrl + scroll = vertical scrolling
                    y_range = ylims[1] - ylims[0]
                    scroll_amount = y_range * 0.1  # 10% of current view
                    
                    if event.button == 'up':
                        # Scroll up
                        ax.set_ylim([ylims[0] + scroll_amount, ylims[1] + scroll_amount])
                    else:
                        # Scroll down
                        ax.set_ylim([ylims[0] - scroll_amount, ylims[1] - scroll_amount])
                        
                else:
                    # Regular scroll = zoom
                    zoom_factor = 1.1 if event.button == 'up' else 1/1.1
                    
                    # Get mouse position
                    x_mouse, y_mouse = event.xdata, event.ydata
                    if x_mouse is not None and y_mouse is not None:
                        # Calculate new limits
                        x_range = (xlims[1] - xlims[0]) * zoom_factor
                        y_range = (ylims[1] - ylims[0]) * zoom_factor
                        
                        ax.set_xlim([x_mouse - x_range/2, x_mouse + x_range/2])
                        ax.set_ylim([y_mouse - y_range/2, y_mouse + y_range/2])
                
                self.data_canvas.draw_idle()
        
        # Also add keyboard support for arrow key scrolling
        def on_key_press(event):
            ax = self.data_figure.gca()
            if ax and event.inaxes == ax:
                xlims = ax.get_xlim()
                ylims = ax.get_ylim()
                x_range = xlims[1] - xlims[0]
                y_range = ylims[1] - ylims[0]
                scroll_amount = 0.1  # 10% of current view
                
                if event.key == 'left':
                    # Scroll left
                    shift = x_range * scroll_amount
                    ax.set_xlim([xlims[0] - shift, xlims[1] - shift])
                    self.data_canvas.draw_idle()
                elif event.key == 'right':
                    # Scroll right
                    shift = x_range * scroll_amount
                    ax.set_xlim([xlims[0] + shift, xlims[1] + shift])
                    self.data_canvas.draw_idle()
                elif event.key == 'up':
                    # Scroll up
                    shift = y_range * scroll_amount
                    ax.set_ylim([ylims[0] + shift, ylims[1] + shift])
                    self.data_canvas.draw_idle()
                elif event.key == 'down':
                    # Scroll down
                    shift = y_range * scroll_amount
                    ax.set_ylim([ylims[0] - shift, ylims[1] - shift])
                    self.data_canvas.draw_idle()
        
        self.data_canvas.mpl_connect('scroll_event', on_scroll)
        self.data_canvas.mpl_connect('key_press_event', on_key_press)
        
        # Make sure canvas can receive keyboard focus
        self.data_canvas.get_tk_widget().focus_set()
        self.data_canvas.get_tk_widget().bind('<Button-1>', lambda e: self.data_canvas.get_tk_widget().focus_set())
        
        # Channel controls and actions (right side)
        controls_frame = ttk.LabelFrame(data_main, text="Controls")
        controls_frame.grid(row=0, column=1, sticky="ns")
        
        # Actions section
        actions_section = ttk.LabelFrame(controls_frame, text="Actions")
        actions_section.pack(fill=tk.X, padx=4, pady=4)
        
        ttk.Button(actions_section, text="Generate Data", command=self.on_generate, width=12).pack(pady=2)
        ttk.Button(actions_section, text="Run Unmix", command=self.on_unmix, width=12).pack(pady=2)
        
        # Channel visibility section
        ch_section = ttk.LabelFrame(controls_frame, text="Channel Visibility")
        ch_section.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        self.channel_checks_frame = ttk.Frame(ch_section)
        self.channel_checks_frame.pack(padx=4, pady=4)
        
        ch_btn_frame = ttk.Frame(ch_section)
        ch_btn_frame.pack(padx=4, pady=4)
        ttk.Button(ch_btn_frame, text="All", command=self._select_all_channels, width=8).pack(pady=(0,2))
        ttk.Button(ch_btn_frame, text="None", command=self._select_no_channels, width=8).pack()
        
        # Unmix tab figure and output
        self.unmix_figure = Figure(figsize=(10, 6), dpi=100)
        self.unmix_canvas = FigureCanvasTkAgg(self.unmix_figure, master=self.tab_unmix)
        self.unmix_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_bottom_panel(self):
        """Build the bottom panel with actions, plots, and output."""
        # Bottom panel for Plots and Output (no actions anymore)
        bottom_panel = ttk.Frame(self)
        bottom_panel.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=8, pady=8)
        bottom_panel.columnconfigure(0, weight=1)
        bottom_panel.columnconfigure(1, weight=1)
        bottom_panel.columnconfigure(2, weight=1)
        
        # Spectral plots panel (left)
        spectral_frame = ttk.LabelFrame(bottom_panel, text="Spectral Profiles")
        spectral_frame.grid(row=0, column=0, sticky="nsew", padx=(0,4))
        
        # Spectral plot controls with fluorophore toggles and expand button
        spectral_controls = ttk.Frame(spectral_frame)
        spectral_controls.pack(fill=tk.X, padx=4, pady=2)
        
        # Fluorophore visibility for spectral profiles (separate from ground truth)
        spectral_fluor_frame = ttk.Frame(spectral_controls)
        spectral_fluor_frame.pack(side=tk.LEFT)
        ttk.Label(spectral_fluor_frame, text="Show:", font=('TkDefaultFont', 8)).pack(side=tk.LEFT, padx=(0,4))
        
        self.spectral_fluor_frame = ttk.Frame(spectral_fluor_frame)
        self.spectral_fluor_frame.pack(side=tk.LEFT)
        
        ttk.Button(spectral_controls, text="🔍 Expand", command=self._expand_spectral_view, width=12).pack(side=tk.RIGHT)
        
        self.spectral_figure = Figure(figsize=(6, 3), dpi=80)
        self.spectral_canvas = FigureCanvasTkAgg(self.spectral_figure, master=spectral_frame)
        self.spectral_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        # Abundance plots panel (middle)
        abundance_frame = ttk.LabelFrame(bottom_panel, text="Ground Truth")
        abundance_frame.grid(row=0, column=1, sticky="nsew", padx=4)
        
        # Abundance plot with fluorophore selection and expand button
        abundance_controls = ttk.Frame(abundance_frame)
        abundance_controls.pack(fill=tk.X, padx=4, pady=2)
        
        # Fluorophore selection dropdown
        ttk.Label(abundance_controls, text="Show:").pack(side=tk.LEFT)
        self.selected_fluorophore = tk.StringVar(value="F1")
        self.fluor_dropdown = ttk.Combobox(abundance_controls, textvariable=self.selected_fluorophore, 
                                          values=["F1", "F2", "F3"], state="readonly", width=8)
        self.fluor_dropdown.pack(side=tk.LEFT, padx=(4,8))
        self.fluor_dropdown.bind("<<ComboboxSelected>>", self._on_fluorophore_change)
        
        ttk.Button(abundance_controls, text="🔍 Expand", command=self._expand_abundance_view, width=12).pack(side=tk.RIGHT)
        
        self.abundance_figure = Figure(figsize=(4, 3), dpi=80)
        self.abundance_canvas = FigureCanvasTkAgg(self.abundance_figure, master=abundance_frame)
        self.abundance_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        # Output terminal (right side of bottom) with hide/show functionality
        self.output_frame = ttk.LabelFrame(bottom_panel, text="Output")
        self.output_frame.grid(row=0, column=2, sticky="nsew", padx=(4,0))
        
        # Output controls
        output_controls = ttk.Frame(self.output_frame)
        output_controls.pack(fill=tk.X, padx=4, pady=2)
        self.terminal_visible = tk.BooleanVar(value=True)
        hide_btn = ttk.Button(output_controls, text="Hide Terminal", command=self._toggle_terminal)
        hide_btn.pack(side=tk.RIGHT)
        
        # Terminal content
        self.terminal_content = ttk.Frame(self.output_frame)
        self.terminal_content.pack(fill=tk.BOTH, expand=True)
        
        self.output = tk.Text(self.terminal_content, height=6)
        output_scroll = ttk.Scrollbar(self.terminal_content, orient="vertical", command=self.output.yview)
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
        
        # Image Dimensions
        dims_group = ttk.LabelFrame(self.settings_frame, text="Image Dimensions")
        dims_group.grid(row=row, column=0, sticky="ew", padx=2, pady=1)
        dims_group.columnconfigure(0, weight=1)
        self.settings_panels['dimensions'] = ImageDimensionsPanel(dims_group)
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
        dims = self.settings_panels['dimensions'].get_dimensions()
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
            
    def _toggle_terminal(self) -> None:
        """Toggle terminal visibility."""
        if self.terminal_visible.get():
            # Hide terminal
            self.terminal_content.pack_forget()
            self.output_frame.config(text="Output (Hidden)")
            # Update button text
            for child in self.output_frame.winfo_children():
                if isinstance(child, ttk.Frame):
                    for btn in child.winfo_children():
                        if isinstance(btn, ttk.Button):
                            btn.config(text="Show Terminal")
                            break
            self.terminal_visible.set(False)
        else:
            # Show terminal
            self.terminal_content.pack(fill=tk.BOTH, expand=True)
            self.output_frame.config(text="Output")
            # Update button text
            for child in self.output_frame.winfo_children():
                if isinstance(child, ttk.Frame):
                    for btn in child.winfo_children():
                        if isinstance(btn, ttk.Button):
                            btn.config(text="Hide Terminal")
                            break
            self.terminal_visible.set(True)

    def on_generate(self) -> None:
        """Generate synthetic data based on current settings."""
        try:
            # Get configurations from all panels
            seed = self.settings_panels['seed'].get_seed()
            rng = np.random.default_rng(seed)
            
            grid_config = self.settings_panels['grid'].get_wavelength_grid()
            ch_config = self.settings_panels['channels'].get_channel_config()
            dims_config = self.settings_panels['dimensions'].get_dimensions()
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
            H, W = dims_config['H'], dims_config['W']
            px_nm = dims_config['pixel_nm']
            field = FieldSpec(shape=(H, W), pixel_size_nm=px_nm)
            
            # Generate abundance field
            af = AbundanceField(rng)
            
            # If objects exist, build from objects only
            objects = self.object_manager.get_objects()
            if len(objects) > 0:
                A = af.build_from_objects(K=K, field=field, objects=objects, base=None)
            else:
                # Default uniform field when no objects are defined
                A = af.sample(K=K, field=field, kind="uniform")

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
            self._populate_fluor_dropdown(K)
            self._populate_spectral_fluor_toggles(K)
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
            ttk.Checkbutton(self.channel_checks_frame, text=f"C{i+1}", variable=var, command=self._render_data_view).pack(anchor="w", pady=1)

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

    def _populate_fluor_dropdown(self, K: int) -> None:
        """Populate fluorophore dropdown options."""
        fluor_options = [f"F{i+1}" for i in range(K)]
        self.fluor_dropdown['values'] = fluor_options
        if fluor_options:
            self.selected_fluorophore.set(fluor_options[0])
        else:
            self.selected_fluorophore.set("")
            
    def _on_fluorophore_change(self, event=None):
        """Handle fluorophore selection change."""
        self._render_data_view()
        
    def _populate_spectral_fluor_toggles(self, K: int) -> None:
        """Populate fluorophore visibility toggles for spectral profiles."""
        for w in self.spectral_fluor_frame.winfo_children():
            w.destroy()
        self.spectral_fluor_vars = []
        for i in range(K):
            var = tk.BooleanVar(value=True)
            self.spectral_fluor_vars.append(var)
            ttk.Checkbutton(self.spectral_fluor_frame, text=f"F{i+1}", variable=var, 
                           command=self._render_data_view).pack(side=tk.LEFT, padx=2)

    def _render_data_view(self) -> None:
        """Render the data visualization."""
        if self.current_Y is None or self.current_field is None or self.current_spectral is None:
            return
        H, W = self.current_field.shape
        L, P = self.current_Y.shape
        active_ch = [i for i, v in enumerate(self.channel_vars) if v.get()]
        
        # Get selected fluorophore index from dropdown
        selected_fluor_str = self.selected_fluorophore.get()
        if selected_fluor_str and selected_fluor_str.startswith('F'):
            try:
                selected_fluor_idx = int(selected_fluor_str[1:]) - 1  # Convert F1 -> 0, F2 -> 1, etc.
                active_fl = [selected_fluor_idx] if 0 <= selected_fluor_idx < self.current_A.shape[0] else []
            except (ValueError, IndexError):
                active_fl = []
        else:
            active_fl = []
        
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

        # Clear and create main composite image view (full screen)
        self.data_figure.clear()
        
        # Main composite image takes the entire space
        ax_img = self.data_figure.add_subplot(1, 1, 1)
        ax_img.imshow(rgb)
        ax_img.set_title("Composite Image (Selected Channels)", fontsize=14)
        ax_img.axis("off")
        
        # Get active fluorophores for spectral profiles (separate from ground truth)
        spectral_active_fl = [i for i, v in enumerate(self.spectral_fluor_vars) if v.get()] if hasattr(self, 'spectral_fluor_vars') else []
        
        # Render spectral and abundance plots in bottom panels
        self._render_spectral_panel(active_ch, spectral_active_fl)
        self._render_abundance_maps(active_fl, H, W)
        
        self.data_canvas.draw_idle()

    def _render_spectral_panel(self, active_ch, active_fl):
        """Render the spectral profiles panel."""
        self.spectral_figure.clear()
        ax_spec = self.spectral_figure.add_subplot(1, 1, 1)
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
        
        self.spectral_canvas.draw_idle()

    def _render_abundance_maps(self, active_fl, H, W):
        """Render abundance maps for selected fluorophores."""
        self.abundance_figure.clear()
        if self.current_A is not None and len(active_fl) > 0:
            ax_maps = self.abundance_figure.add_subplot(1, 1, 1)
            n_show = min(len(active_fl), 1)  # Show only one at a time for clarity
            if n_show > 0:
                k = active_fl[0]  # Show first selected
                im = ax_maps.imshow(self.current_A[k].reshape(H, W), cmap="magma")
                ax_maps.set_title(f"Ground Truth: F{k+1} Abundance", fontsize=10)
                ax_maps.axis("off")
                self.abundance_figure.colorbar(im, ax=ax_maps, fraction=0.046, pad=0.04)
        else:
            # Show placeholder when no fluorophores are selected
            ax_maps = self.abundance_figure.add_subplot(1, 1, 1)
            ax_maps.text(0.5, 0.5, 'Select fluorophores\nto view abundance', 
                        ha='center', va='center', transform=ax_maps.transAxes,
                        fontsize=10, color='gray')
            ax_maps.axis("off")
        
        self.abundance_canvas.draw_idle()

    def _expand_spectral_view(self):
        """Open an expanded view of the spectral profiles."""
        if self.current_Y is None or self.current_spectral is None:
            return
            
        # Create a new window for expanded spectral view
        expand_window = tk.Toplevel(self)
        expand_window.title("Expanded Spectral Profiles")
        expand_window.geometry("800x600")
        
        # Create larger figure
        expand_figure = Figure(figsize=(10, 8), dpi=100)
        expand_canvas = FigureCanvasTkAgg(expand_figure, master=expand_window)
        expand_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Render the spectral plot in the expanded view
        active_ch = [i for i, v in enumerate(self.channel_vars) if v.get()]
        # Use the same fluorophore visibility as the main spectral view
        active_fl = [i for i, v in enumerate(self.spectral_fluor_vars) if v.get()] if hasattr(self, 'spectral_fluor_vars') else []
        
        ax_spec = expand_figure.add_subplot(1, 1, 1)
        grid_config = self.settings_panels['grid'].get_wavelength_grid()
        start = grid_config['start']
        stop = grid_config['stop']
        ax_spec.set_xlim(start, stop)
        ax_spec.set_ylim(0, 1.1)
        ax_spec.set_xlabel("Wavelength (nm)", fontsize=12)
        ax_spec.set_ylabel("Normalized Intensity", fontsize=12)
        ax_spec.set_title("Spectral Profiles (Expanded View)", fontsize=14)
        
        # Replicate the spectral plotting logic from _render_spectral_panel
        # Draw channel bands
        for idx, ch in enumerate(self.current_spectral.channels):
            half = 0.5 * ch.bandwidth_nm
            low, high = ch.center_nm - half, ch.center_nm + half
            active_flag = idx in active_ch
            if active_flag:
                face = self._wavelength_to_rgb_nm(ch.center_nm)
                ax_spec.axvspan(low, high, ymin=0, ymax=0.15, facecolor=face, alpha=0.7, edgecolor='k', linewidth=1)
                ax_spec.text(ch.center_nm, 0.08, f"C{idx+1}", ha='center', va='center', fontsize=10, weight='bold')
            else:
                ax_spec.axvspan(low, high, ymin=0, ymax=0.15, facecolor='lightgray', alpha=0.3, edgecolor='gray')
                
        # Show fluorophore profiles and measured signal
        if self.current_A is not None and self.current_M is not None:
            K = self.current_A.shape[0]
            
            for k in range(K):
                show = k in active_fl
                if show:
                    pdf = self.current_spectral._pdf(self.current_spectral.fluors[k])
                    pdf_norm = pdf / (np.max(pdf) + 1e-9)
                    peak_nm = float(self.current_spectral.lambdas[np.argmax(pdf)])
                    col = self._wavelength_to_rgb_nm(peak_nm)
                    ax_spec.plot(self.current_spectral.lambdas, 0.2 + 0.6 * pdf_norm, 
                               color=col, linewidth=3, label=f"F{k+1} (theory)", alpha=0.8)
            
            # Measured total signal
            total_per_channel = np.sum(self.current_Y, axis=1)
            channel_centers = np.array([ch.center_nm for ch in self.current_spectral.channels])
            
            if len(channel_centers) >= 2:
                interp = np.interp(self.current_spectral.lambdas, channel_centers, total_per_channel, 
                                   left=total_per_channel[0], right=total_per_channel[-1])
            else:
                interp = np.full_like(self.current_spectral.lambdas, total_per_channel[0] if len(total_per_channel) > 0 else 0.0)

            lambdas = self.current_spectral.lambdas
            if len(lambdas) > 3:
                step_nm = max(1.0, float(np.median(np.diff(lambdas))))
                window = max(3, int(round(5.0 / step_nm)) | 1)
                kernel = np.ones(window, dtype=np.float32)
                kernel /= np.sum(kernel)
                interp_smooth = np.convolve(interp, kernel, mode='same')
            else:
                interp_smooth = interp

            if np.max(interp_smooth) > 0:
                curve = 0.2 + 0.6 * (interp_smooth / np.max(interp_smooth))
                ax_spec.plot(self.current_spectral.lambdas, curve, color='black', linewidth=3, label='Measured Total (cont.)', alpha=0.9)
            else:
                ax_spec.plot(self.current_spectral.lambdas, np.zeros_like(self.current_spectral.lambdas) + 0.2, color='black', linewidth=3, label='Measured Total (cont.)', alpha=0.9)
                           
            ax_spec.legend(loc='upper right', fontsize=12, framealpha=0.8)
        
        expand_canvas.draw()

    def _expand_abundance_view(self):
        """Open an expanded view of the abundance maps."""
        if self.current_A is None:
            return
            
        # Show all fluorophores in expanded view
        K = self.current_A.shape[0]
        active_fl = list(range(K))
            
        # Create a new window for expanded abundance view
        expand_window = tk.Toplevel(self)
        expand_window.title("Expanded Ground Truth Abundance Maps")
        expand_window.geometry("1000x700")
        
        # Create larger figure with subplots for multiple fluorophores
        expand_figure = Figure(figsize=(12, 8), dpi=100)
        expand_canvas = FigureCanvasTkAgg(expand_figure, master=expand_window)
        expand_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        H, W = self.current_field.shape
        n_fl = len(active_fl)
        
        # Calculate subplot layout
        if n_fl == 1:
            rows, cols = 1, 1
        elif n_fl <= 2:
            rows, cols = 1, 2
        elif n_fl <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 2, 3  # Show max 6 fluorophores
        
        for i, k in enumerate(active_fl[:6]):  # Limit to 6 fluorophores
            ax = expand_figure.add_subplot(rows, cols, i + 1)
            im = ax.imshow(self.current_A[k].reshape(H, W), cmap="magma")
            ax.set_title(f"F{k+1} Abundance Map", fontsize=12)
            ax.axis("off")
            expand_figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        expand_figure.suptitle("Ground Truth Abundance Maps (Expanded View)", fontsize=16)
        expand_figure.tight_layout()
        expand_canvas.draw()

    def _expand_composite_view(self):
        """Open an expanded view of the composite image."""
        if self.current_Y is None or self.current_field is None or self.current_spectral is None:
            return
            
        # Create a new window for expanded composite view
        expand_window = tk.Toplevel(self)
        expand_window.title("Expanded Composite Image")
        expand_window.geometry("1200x900")
        
        # Create larger figure with navigation
        expand_figure = Figure(figsize=(14, 10), dpi=100)
        expand_figure.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05)
        expand_canvas = FigureCanvasTkAgg(expand_figure, master=expand_window)
        expand_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add navigation toolbar
        expand_toolbar = NavigationToolbar2Tk(expand_canvas, expand_window)
        expand_toolbar.update()
        
        # Add the same scrolling functionality to expanded view
        def on_expand_scroll(event):
            ax = expand_figure.gca()
            if ax and event.inaxes == ax:
                xlims = ax.get_xlim()
                ylims = ax.get_ylim()
                
                if event.key == 'shift':
                    # Horizontal scrolling
                    x_range = xlims[1] - xlims[0]
                    scroll_amount = x_range * 0.1
                    if event.button == 'up':
                        ax.set_xlim([xlims[0] - scroll_amount, xlims[1] - scroll_amount])
                    else:
                        ax.set_xlim([xlims[0] + scroll_amount, xlims[1] + scroll_amount])
                elif event.key == 'control':
                    # Vertical scrolling
                    y_range = ylims[1] - ylims[0]
                    scroll_amount = y_range * 0.1
                    if event.button == 'up':
                        ax.set_ylim([ylims[0] + scroll_amount, ylims[1] + scroll_amount])
                    else:
                        ax.set_ylim([ylims[0] - scroll_amount, ylims[1] - scroll_amount])
                else:
                    # Zoom
                    zoom_factor = 1.1 if event.button == 'up' else 1/1.1
                    x_mouse, y_mouse = event.xdata, event.ydata
                    if x_mouse is not None and y_mouse is not None:
                        x_range = (xlims[1] - xlims[0]) * zoom_factor
                        y_range = (ylims[1] - ylims[0]) * zoom_factor
                        ax.set_xlim([x_mouse - x_range/2, x_mouse + x_range/2])
                        ax.set_ylim([y_mouse - y_range/2, y_mouse + y_range/2])
                expand_canvas.draw_idle()
        
        expand_canvas.mpl_connect('scroll_event', on_expand_scroll)
        
        # Recreate the composite image
        H, W = self.current_field.shape
        active_ch = [i for i, v in enumerate(self.channel_vars) if v.get()]
        
        # Build RGB composite (same logic as main view)
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
        
        # Display in expanded window
        ax = expand_figure.add_subplot(1, 1, 1)
        ax.imshow(rgb)
        ax.set_title("Composite Image (Selected Channels) - Expanded View", fontsize=16)
        ax.axis("off")
        
        expand_canvas.draw()

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
