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


class FluorophoreEditor(ttk.Frame):
    def __init__(self, parent, fluor_idx, on_update_callback, initial_data=None):
        super().__init__(parent)
        self.fluor_idx = fluor_idx
        self.on_update = on_update_callback
        
        # Load initial data
        if initial_data:
            self.data = initial_data.copy()
        else:
            self.data = {
                'name': f'F{fluor_idx+1}',
                'model': 'gaussian',
                'params': {'mu': 520.0, 'sigma': 12.0},
                'brightness': 1.0
            }
        
        # Model selection
        model_frame = ttk.Frame(self)
        model_frame.pack(fill=tk.X, pady=2)
        ttk.Label(model_frame, text="Model:").grid(row=0, column=0, sticky="w", padx=(0,4))
        self.model_var = tk.StringVar(value=self.data['model'])
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                 values=["gaussian", "skewnorm", "lognormal", "weibull"], 
                                 width=12, state="readonly")
        model_combo.grid(row=0, column=1, padx=2)
        model_combo.bind("<<ComboboxSelected>>", self._on_model_change)
        
        # Parameters frame (dynamic based on model)
        self.params_frame = ttk.Frame(self)
        self.params_frame.pack(fill=tk.X, pady=2)
        
        # Brightness
        bright_frame = ttk.Frame(self)
        bright_frame.pack(fill=tk.X, pady=2)
        ttk.Label(bright_frame, text="Brightness:").grid(row=0, column=0, sticky="w", padx=(0,4))
        self.brightness_var = tk.DoubleVar(value=self.data['brightness'])
        brightness_entry = ttk.Entry(bright_frame, textvariable=self.brightness_var, width=8)
        brightness_entry.grid(row=0, column=1, padx=2)
        brightness_entry.bind('<KeyRelease>', self._on_param_change)
        
        self.param_vars = {}
        self._build_params()
        
    def _on_model_change(self, event=None):
        self.data['model'] = self.model_var.get()
        self._build_params()
        self._update_data()
        
    def _on_param_change(self, event=None):
        self._update_data()
            
    def _build_params(self):
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        self.param_vars.clear()
        
        model = self.model_var.get()
        current_params = self.data.get('params', {})
        
        if model == "gaussian":
            self._add_param("μ (nm)", "mu", current_params.get('mu', 520.0))
            self._add_param("σ (nm)", "sigma", current_params.get('sigma', 12.0))
        elif model == "skewnorm":
            self._add_param("μ (nm)", "mu", current_params.get('mu', 520.0))
            self._add_param("σ (nm)", "sigma", current_params.get('sigma', 12.0))
            self._add_param("α", "alpha", current_params.get('alpha', 4.0))
        elif model == "lognormal":
            self._add_param("log μ", "mu", current_params.get('mu', 6.2))
            self._add_param("log σ", "sigma", current_params.get('sigma', 0.08))
        elif model == "weibull":
            self._add_param("k (shape)", "k", current_params.get('k', 2.0))
            self._add_param("λ (scale)", "lam", current_params.get('lam', 20.0))
            self._add_param("shift (nm)", "shift", current_params.get('shift', 500.0))
            
    def _add_param(self, label, key, default_val):
        row = len(self.param_vars)
        ttk.Label(self.params_frame, text=label).grid(row=row, column=0, sticky="w", padx=(0,4))
        var = tk.DoubleVar(value=default_val)
        self.param_vars[key] = var
        entry = ttk.Entry(self.params_frame, textvariable=var, width=8)
        entry.grid(row=row, column=1, padx=2, pady=1)
        entry.bind('<KeyRelease>', self._on_param_change)
        
    def _update_data(self):
        self.data['model'] = self.model_var.get()
        self.data['brightness'] = self.brightness_var.get()
        self.data['params'] = {k: v.get() for k, v in self.param_vars.items()}
        if self.on_update:
            self.on_update(self.fluor_idx, self.data)
        
    def get_fluorophore(self):
        return Fluorophore(
            name=self.data['name'],
            model=self.data['model'],
            params=self.data['params'],
            brightness=self.data['brightness']
        )


class PlaygroundGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Spectral Unmixing Playground")
        # State
        self.current_Y = None
        self.current_A = None
        self.current_B = None
        self.current_M = None
        self.current_field = None
        self.current_spectral = None
        self.channel_vars = []
        self.fluor_vars = []
        self.fluor_editors = []
        self.settings_visible = True
        self.objects = []  # object layers for spatial placement
        self.show_regions = tk.BooleanVar(value=False)
        self._build_widgets()

    def _build_widgets(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

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
        row = 0
        
        # Grid settings group (compact)
        grid_group = ttk.LabelFrame(self.settings_frame, text="Wavelength Grid")
        grid_group.grid(row=row, column=0, sticky="ew", padx=2, pady=1)
        grid_group.columnconfigure(0, weight=1)
        row += 1
        
        grid_frame = ttk.Frame(grid_group)
        grid_frame.pack(fill=tk.X, padx=2, pady=2)
        
        ttk.Label(grid_frame, text="Start").grid(row=0, column=0, sticky="w")
        ttk.Label(grid_frame, text="Stop").grid(row=0, column=1, sticky="w")
        ttk.Label(grid_frame, text="Step").grid(row=0, column=2, sticky="w")
        
        self.grid_start = tk.DoubleVar(value=450.0)
        self.grid_stop = tk.DoubleVar(value=700.0)
        self.grid_step = tk.DoubleVar(value=1.0)
        ttk.Entry(grid_frame, textvariable=self.grid_start, width=8).grid(row=1, column=0, padx=(0,2))
        ttk.Entry(grid_frame, textvariable=self.grid_stop, width=8).grid(row=1, column=1, padx=(0,2))
        ttk.Entry(grid_frame, textvariable=self.grid_step, width=8).grid(row=1, column=2)
        
        # Channels group (compact)
        ch_group = ttk.LabelFrame(self.settings_frame, text="Detection Channels")
        ch_group.grid(row=row, column=0, sticky="ew", padx=2, pady=1)
        ch_group.columnconfigure(0, weight=1)
        row += 1
        
        ch_frame = ttk.Frame(ch_group)
        ch_frame.pack(fill=tk.X, padx=2, pady=2)
        
        ttk.Label(ch_frame, text="Count (L)").grid(row=0, column=0, sticky="w")
        ttk.Label(ch_frame, text="Bandwidth (nm)").grid(row=0, column=1, sticky="w")
        
        self.num_channels = tk.IntVar(value=4)
        self.bandwidth = tk.DoubleVar(value=30.0)
        ttk.Entry(ch_frame, textvariable=self.num_channels, width=8).grid(row=1, column=0, padx=(0,2))
        ttk.Entry(ch_frame, textvariable=self.bandwidth, width=12).grid(row=1, column=1)
        
        # Fluorophores group (compact)
        fluor_group = ttk.LabelFrame(self.settings_frame, text="Fluorophores")
        fluor_group.grid(row=row, column=0, sticky="ew", padx=2, pady=1)
        fluor_group.columnconfigure(0, weight=1)
        row += 1
        
        # Fluorophore list with treeview
        fluor_main_frame = ttk.Frame(fluor_group)
        fluor_main_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # List header with controls
        list_header = ttk.Frame(fluor_main_frame)
        list_header.pack(fill=tk.X, pady=(0,4))
        ttk.Label(list_header, text="Fluorophore List:", font=('TkDefaultFont', 9, 'bold')).pack(side=tk.LEFT)
        
        btn_frame = ttk.Frame(list_header)
        btn_frame.pack(side=tk.RIGHT)
        ttk.Button(btn_frame, text="+ Add", command=self._add_fluorophore, width=8).pack(side=tk.LEFT, padx=(0,2))
        ttk.Button(btn_frame, text="- Remove", command=self._remove_fluorophore, width=8).pack(side=tk.LEFT)
        
        # Treeview for fluorophore list (more compact)
        tree_frame = ttk.Frame(fluor_main_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.fluor_tree = ttk.Treeview(tree_frame, columns=('model', 'params'), show='tree headings', height=4)
        self.fluor_tree.heading('#0', text='Name', anchor='w')
        self.fluor_tree.heading('model', text='Model', anchor='w')
        self.fluor_tree.heading('params', text='Key Parameters', anchor='w')
        
        self.fluor_tree.column('#0', width=50, minwidth=50)
        self.fluor_tree.column('model', width=70, minwidth=70)
        self.fluor_tree.column('params', width=150, minwidth=120)
        
        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.fluor_tree.yview)
        self.fluor_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.fluor_tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")
        
        # Bind selection event
        self.fluor_tree.bind('<<TreeviewSelect>>', self._on_fluor_select)
        
        # Editor panel for selected fluorophore
        editor_label = ttk.Label(fluor_main_frame, text="Edit Selected Fluorophore:", font=('TkDefaultFont', 9, 'bold'))
        editor_label.pack(anchor='w', pady=(8,2))
        
        self.fluor_editor_frame = ttk.Frame(fluor_main_frame, relief='groove', borderwidth=2)
        self.fluor_editor_frame.pack(fill=tk.X, pady=2)
        
        self.current_editor = None
        self.fluor_data = []  # Store fluorophore data
        
        # Initialize with 3 fluorophores
        self._add_fluorophore()
        self._add_fluorophore()
        self._add_fluorophore()
        
        # Spatial group (compact)
        spatial_group = ttk.LabelFrame(self.settings_frame, text="Spatial Field")
        spatial_group.grid(row=row, column=0, sticky="ew", padx=2, pady=1)
        spatial_group.columnconfigure(0, weight=1)
        row += 1
        
        # Main spatial container using grid throughout
        spatial_main = ttk.Frame(spatial_group)
        spatial_main.pack(fill=tk.X, padx=2, pady=2)
        
        # Basic dimensions frame
        dims_frame = ttk.Frame(spatial_main)
        dims_frame.pack(fill=tk.X, pady=(0,4))
        
        ttk.Label(dims_frame, text="H").grid(row=0, column=0, sticky="w")
        ttk.Label(dims_frame, text="W").grid(row=0, column=1, sticky="w")
        ttk.Label(dims_frame, text="Pixel (nm)").grid(row=0, column=2, sticky="w")
        
        self.H = tk.IntVar(value=128)
        self.W = tk.IntVar(value=128)
        self.pixel_nm = tk.DoubleVar(value=100.0)
        ttk.Entry(dims_frame, textvariable=self.H, width=6).grid(row=1, column=0, padx=(0,2))
        ttk.Entry(dims_frame, textvariable=self.W, width=6).grid(row=1, column=1, padx=(0,2))
        ttk.Entry(dims_frame, textvariable=self.pixel_nm, width=8).grid(row=1, column=2)
        
        # Global field parameters (compact layout)
        global_frame = ttk.LabelFrame(spatial_main, text="Global Field Settings")
        global_frame.pack(fill=tk.X, pady=(4,0))
        global_frame.columnconfigure((0,1,2), weight=1)
        
        # Row 0: Type, Density, Spot σ
        ttk.Label(global_frame, text="Type:").grid(row=0, column=0, sticky="w", padx=2)
        self.spatial_kind = tk.StringVar(value="dots")
        kind_combo = ttk.Combobox(global_frame, textvariable=self.spatial_kind,
                                  values=["dots", "uniform", "circles", "boxes", "gaussian_blobs", "mixed"],
                                  state="readonly", width=10)
        kind_combo.grid(row=1, column=0, sticky="ew", padx=2)
        
        ttk.Label(global_frame, text="Density (/100×100μm²):").grid(row=0, column=1, sticky="w", padx=2)
        self.density = tk.DoubleVar(value=50.0)
        ttk.Entry(global_frame, textvariable=self.density, width=8).grid(row=1, column=1, sticky="ew", padx=2)
        
        ttk.Label(global_frame, text="Spot σ (px):").grid(row=0, column=2, sticky="w", padx=2)
        self.spot_sigma = tk.DoubleVar(value=1.2)
        ttk.Entry(global_frame, textvariable=self.spot_sigma, width=8).grid(row=1, column=2, sticky="ew", padx=2)

        # Row 1: Count, Size, Intensity
        ttk.Label(global_frame, text="Count/fluor:").grid(row=2, column=0, sticky="w", padx=2, pady=(4,0))
        self.count_per_fluor = tk.IntVar(value=50)
        ttk.Entry(global_frame, textvariable=self.count_per_fluor, width=8).grid(row=3, column=0, sticky="ew", padx=2)

        ttk.Label(global_frame, text="Size (px):").grid(row=2, column=1, sticky="w", padx=2, pady=(4,0))
        self.size_px = tk.DoubleVar(value=6.0)
        ttk.Entry(global_frame, textvariable=self.size_px, width=8).grid(row=3, column=1, sticky="ew", padx=2)

        # Intensity range in one column
        ttk.Label(global_frame, text="Intensity min-max:").grid(row=2, column=2, sticky="w", padx=2, pady=(4,0))
        intensity_frame = ttk.Frame(global_frame)
        intensity_frame.grid(row=3, column=2, sticky="ew", padx=2)
        self.intensity_min = tk.DoubleVar(value=0.5)
        self.intensity_max = tk.DoubleVar(value=1.5)
        ttk.Entry(intensity_frame, textvariable=self.intensity_min, width=4).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(intensity_frame, text="-").pack(side=tk.LEFT)
        ttk.Entry(intensity_frame, textvariable=self.intensity_max, width=4).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Object Layers group (compact)
        objects_group = ttk.LabelFrame(self.settings_frame, text="Object Layers")
        objects_group.grid(row=row, column=0, sticky="ew", padx=2, pady=1)
        objects_group.columnconfigure(0, weight=1)
        row += 1

        # Quick toggle and help
        toggle_frame = ttk.Frame(objects_group)
        toggle_frame.pack(fill=tk.X, padx=4, pady=(4,0))
        self.include_base_field = tk.BooleanVar(value=True)
        ttk.Checkbutton(toggle_frame, text="Include Global Field", variable=self.include_base_field).pack(side=tk.LEFT)
        ttk.Label(toggle_frame, text="(Place specific fluorophores in regions)", foreground="gray", font=('TkDefaultFont', 8)).pack(side=tk.RIGHT)

        obj_main = ttk.Frame(objects_group)
        obj_main.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        obj_main.columnconfigure(0, weight=1)

        # Object list with improved buttons
        list_frame = ttk.Frame(obj_main)
        list_frame.grid(row=0, column=0, sticky="nsew", pady=(0,4))
        list_frame.columnconfigure(0, weight=1)
        
        header_frame = ttk.Frame(list_frame)
        header_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0,4))
        header_frame.columnconfigure(0, weight=1)
        ttk.Label(header_frame, text="Objects", font=('TkDefaultFont', 9, 'bold')).pack(side=tk.LEFT)
        
        btns = ttk.Frame(header_frame)
        btns.pack(side=tk.RIGHT)
        ttk.Button(btns, text="Add", width=6, command=self._add_object).pack(side=tk.LEFT, padx=(0,2))
        ttk.Button(btns, text="Remove", width=6, command=self._remove_object).pack(side=tk.LEFT, padx=(0,2))
        ttk.Button(btns, text="Copy", width=6, command=self._duplicate_object).pack(side=tk.LEFT)

        # Simplified object list
        self.obj_tree = ttk.Treeview(list_frame, columns=("fluor","kind","region","count"), show='headings', height=4)
        for col, text, w in [("fluor","Fluor",50),("kind","Kind",80),("region","Region",80),("count","Count",50)]:
            self.obj_tree.heading(col, text=text, anchor='w')
            self.obj_tree.column(col, width=w, stretch=True, anchor='w')
        self.obj_tree.grid(row=1, column=0, sticky='nsew')
        obj_scroll = ttk.Scrollbar(list_frame, orient='vertical', command=self.obj_tree.yview)
        self.obj_tree.configure(yscrollcommand=obj_scroll.set)
        obj_scroll.grid(row=1, column=1, sticky='ns')
        self.obj_tree.bind('<<TreeviewSelect>>', self._on_object_select)

        # Compact editor with tabs
        editor_notebook = ttk.Notebook(obj_main)
        editor_notebook.grid(row=1, column=0, sticky='ew', pady=(4,0))
        
        # Basic properties tab
        basic_tab = ttk.Frame(editor_notebook)
        editor_notebook.add(basic_tab, text="Properties")
        
        # Row 0: Fluor and Kind
        ttk.Label(basic_tab, text="Fluorophore:").grid(row=0, column=0, sticky='w', padx=(0,4))
        self.obj_fluor = tk.IntVar(value=0)
        fluor_frame = ttk.Frame(basic_tab)
        fluor_frame.grid(row=0, column=1, sticky='w')
        ttk.Entry(fluor_frame, textvariable=self.obj_fluor, width=4).pack(side=tk.LEFT)
        ttk.Label(fluor_frame, text="(0-based)", foreground="gray", font=('TkDefaultFont', 8)).pack(side=tk.LEFT, padx=(2,0))
        
        ttk.Label(basic_tab, text="Kind:").grid(row=0, column=2, sticky='w', padx=(12,4))
        self.obj_kind = tk.StringVar(value="gaussian_blobs")
        ttk.Combobox(basic_tab, textvariable=self.obj_kind, values=["circles","boxes","gaussian_blobs","dots"], state='readonly', width=12).grid(row=0, column=3, sticky='w')

        # Row 1: Count and Size
        ttk.Label(basic_tab, text="Count:").grid(row=1, column=0, sticky='w', pady=(6,0))
        self.obj_count = tk.IntVar(value=50)
        ttk.Entry(basic_tab, textvariable=self.obj_count, width=8).grid(row=1, column=1, sticky='w', pady=(6,0))
        
        ttk.Label(basic_tab, text="Size (px):").grid(row=1, column=2, sticky='w', padx=(12,4), pady=(6,0))
        self.obj_size = tk.DoubleVar(value=6.0)
        ttk.Entry(basic_tab, textvariable=self.obj_size, width=8).grid(row=1, column=3, sticky='w', pady=(6,0))

        # Row 2: Intensity range and spot sigma
        ttk.Label(basic_tab, text="Intensity:").grid(row=2, column=0, sticky='w', pady=(6,0))
        intensity_frame = ttk.Frame(basic_tab)
        intensity_frame.grid(row=2, column=1, columnspan=2, sticky='w', pady=(6,0))
        self.obj_i_min = tk.DoubleVar(value=0.5)
        self.obj_i_max = tk.DoubleVar(value=1.5)
        ttk.Entry(intensity_frame, textvariable=self.obj_i_min, width=6).pack(side=tk.LEFT)
        ttk.Label(intensity_frame, text="to").pack(side=tk.LEFT, padx=2)
        ttk.Entry(intensity_frame, textvariable=self.obj_i_max, width=6).pack(side=tk.LEFT)
        
        ttk.Label(basic_tab, text="Spot σ:").grid(row=2, column=3, sticky='w', padx=(12,4), pady=(6,0))
        self.obj_sigma = tk.DoubleVar(value=2.0)
        ttk.Entry(basic_tab, textvariable=self.obj_sigma, width=8).grid(row=2, column=4, sticky='w', pady=(6,0))

        # Region tab
        region_tab = ttk.Frame(editor_notebook)
        editor_notebook.add(region_tab, text="Region")
        
        ttk.Label(region_tab, text="Type:").grid(row=0, column=0, sticky='w')
        self.obj_region_type = tk.StringVar(value="full")
        region_combo = ttk.Combobox(region_tab, textvariable=self.obj_region_type, values=["full","rect","circle"], state='readonly', width=12)
        region_combo.grid(row=0, column=1, sticky='w', padx=(0,8))
        region_combo.bind('<<ComboboxSelected>>', self._on_region_type_change)

        # Container for dynamic region parameters
        self.region_params_frame = ttk.Frame(region_tab)
        self.region_params_frame.grid(row=1, column=0, columnspan=4, sticky='ew', pady=(6,0))
        
        # Initialize variables
        self.obj_x0 = tk.IntVar(value=0)
        self.obj_y0 = tk.IntVar(value=0)
        self.obj_w = tk.IntVar(value=64)
        self.obj_h = tk.IntVar(value=64)
        self.obj_cx = tk.DoubleVar(value=64)
        self.obj_cy = tk.DoubleVar(value=64)
        self.obj_r = tk.DoubleVar(value=40)
        
        # Initialize with full region (no parameters)
        self._update_region_ui()

        # Apply button with better styling
        button_frame = ttk.Frame(obj_main)
        button_frame.grid(row=2, column=0, sticky='ew', pady=(8,0))
        self.apply_btn = ttk.Button(button_frame, text="Apply Changes", command=self._apply_object_edits, style='Accent.TButton')
        self.apply_btn.pack(side=tk.LEFT)
        ttk.Label(button_frame, text="Select an object above to edit", foreground="gray", font=('TkDefaultFont', 8)).pack(side=tk.RIGHT)
        
        # Noise group (compact)
        noise_group = ttk.LabelFrame(self.settings_frame, text="Noise Model")
        noise_group.grid(row=row, column=0, sticky="ew", padx=2, pady=1)
        noise_group.columnconfigure(0, weight=1)
        row += 1
        
        noise_main_frame = ttk.Frame(noise_group)
        noise_main_frame.pack(fill=tk.X, padx=2, pady=2)
        
        # Noise enable/disable toggle
        self.noise_enabled = tk.BooleanVar(value=True)
        noise_toggle = ttk.Checkbutton(noise_main_frame, text="Enable Noise", variable=self.noise_enabled, command=self._on_noise_toggle)
        noise_toggle.pack(anchor='w', pady=(0, 4))
        
        # Noise parameters frame
        self.noise_params_frame = ttk.Frame(noise_main_frame)
        self.noise_params_frame.pack(fill=tk.X)
        
        ttk.Label(self.noise_params_frame, text="Gain").grid(row=0, column=0, sticky="w")
        ttk.Label(self.noise_params_frame, text="Read σ").grid(row=0, column=1, sticky="w")
        ttk.Label(self.noise_params_frame, text="Dark rate").grid(row=0, column=2, sticky="w")
        
        self.gain = tk.DoubleVar(value=1.0)
        self.read_sigma = tk.DoubleVar(value=1.0)
        self.dark_rate = tk.DoubleVar(value=0.0)
        self.gain_entry = ttk.Entry(self.noise_params_frame, textvariable=self.gain, width=6)
        self.read_entry = ttk.Entry(self.noise_params_frame, textvariable=self.read_sigma, width=6)
        self.dark_entry = ttk.Entry(self.noise_params_frame, textvariable=self.dark_rate, width=6)
        
        self.gain_entry.grid(row=1, column=0, padx=(0,2))
        self.read_entry.grid(row=1, column=1, padx=(0,2))
        self.dark_entry.grid(row=1, column=2)
        
        # Methods group (compact)
        methods_group = ttk.LabelFrame(self.settings_frame, text="Unmixing Methods")
        methods_group.grid(row=row, column=0, sticky="ew", padx=2, pady=1)
        methods_group.columnconfigure(0, weight=1)
        row += 1
        
        methods_frame = ttk.Frame(methods_group)
        methods_frame.pack(fill=tk.X, padx=2, pady=2)
        
        self.use_nnls = tk.BooleanVar(value=True)
        self.use_lasso = tk.BooleanVar(value=False)
        self.use_nmf = tk.BooleanVar(value=False)
        self.use_em = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(methods_frame, text="NNLS", variable=self.use_nnls).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(methods_frame, text="LASSO", variable=self.use_lasso).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(methods_frame, text="NMF", variable=self.use_nmf).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(methods_frame, text="EM", variable=self.use_em).grid(row=1, column=1, sticky="w")
        
        ttk.Label(methods_frame, text="LASSO α").grid(row=2, column=0, sticky="w", pady=(4,0))
        ttk.Label(methods_frame, text="NMF iters").grid(row=2, column=1, sticky="w", pady=(4,0))
        
        self.lasso_alpha = tk.DoubleVar(value=0.01)
        self.nmf_iters = tk.IntVar(value=200)
        self.em_iters = tk.IntVar(value=150)
        ttk.Entry(methods_frame, textvariable=self.lasso_alpha, width=8).grid(row=3, column=0, padx=(0,2))
        ttk.Entry(methods_frame, textvariable=self.nmf_iters, width=8).grid(row=3, column=1)
        
        # Seed control (compact)
        seed_group = ttk.LabelFrame(self.settings_frame, text="Random Seed")
        seed_group.grid(row=row, column=0, sticky="ew", padx=2, pady=1)
        seed_group.columnconfigure(0, weight=1)
        row += 1
        
        seed_frame = ttk.Frame(seed_group)
        seed_frame.pack(fill=tk.X, padx=2, pady=2)
        
        ttk.Label(seed_frame, text="Seed").grid(row=0, column=0, sticky="w")
        self.seed = tk.IntVar(value=123)
        ttk.Entry(seed_frame, textvariable=self.seed, width=12).grid(row=1, column=0, pady=(2,0))
        


    def _add_fluorophore(self):
        idx = len(self.fluor_data)
        
        # Create default fluorophore data
        default_centers = [480, 520, 560, 600, 650, 700]
        center = default_centers[idx % len(default_centers)]
        
        fluor_data = {
            'name': f'F{idx+1}',
            'model': 'gaussian',
            'params': {'mu': center, 'sigma': 12.0},
            'brightness': 1.0
        }
        
        self.fluor_data.append(fluor_data)
        self._update_fluor_list()
        
        # Select the new item
        item_id = f'fluor_{idx}'
        self.fluor_tree.selection_set(item_id)
        self._on_fluor_select()
        
    def _remove_fluorophore(self):
        if self.fluor_data:
            selected = self.fluor_tree.selection()
            if selected:
                # Remove selected item
                item_id = selected[0]
                idx = int(item_id.split('_')[1])
                self.fluor_data.pop(idx)
                # Update names and indices
                for i, data in enumerate(self.fluor_data):
                    data['name'] = f'F{i+1}'
            else:
                # Remove last item if nothing selected
                self.fluor_data.pop()
            
            self._update_fluor_list()
            if self.current_editor:
                self.current_editor.destroy()
                self.current_editor = None
                
    def _update_fluor_list(self):
        # Clear treeview
        for item in self.fluor_tree.get_children():
            self.fluor_tree.delete(item)
            
        # Populate with current data
        for i, data in enumerate(self.fluor_data):
            params_str = self._format_params(data['model'], data['params'])
            self.fluor_tree.insert('', 'end', iid=f'fluor_{i}', 
                                 text=data['name'], 
                                 values=(data['model'], params_str))
                                 
    def _format_params(self, model, params):
        if model == 'gaussian':
            return f"μ={params.get('mu', 0):.0f}nm, σ={params.get('sigma', 0):.1f}nm"
        elif model == 'skewnorm':
            return f"μ={params.get('mu', 0):.0f}nm, σ={params.get('sigma', 0):.1f}nm, α={params.get('alpha', 0):.1f}"
        elif model == 'lognormal':
            return f"log μ={params.get('mu', 0):.2f}, log σ={params.get('sigma', 0):.3f}"
        elif model == 'weibull':
            return f"k={params.get('k', 0):.1f}, λ={params.get('lam', 0):.1f}, shift={params.get('shift', 0):.0f}nm"
        return str(params)
        
    def _on_fluor_select(self, event=None):
        selected = self.fluor_tree.selection()
        if not selected:
            return
            
        item_id = selected[0]
        idx = int(item_id.split('_')[1])
        
        # Clear previous editor
        if self.current_editor:
            self.current_editor.destroy()
            
        # Create new editor for selected fluorophore
        self.current_editor = FluorophoreEditor(self.fluor_editor_frame, idx, self._on_fluor_update, self.fluor_data[idx])
        self.current_editor.pack(fill=tk.X, padx=4, pady=4)
            
    def _on_fluor_update(self, idx, data):
        # Update the stored data
        if idx < len(self.fluor_data):
            self.fluor_data[idx] = data
            self._update_fluor_list()
            # Reselect the item
            self.fluor_tree.selection_set(f'fluor_{idx}')
            
    def _on_noise_toggle(self):
        # Enable/disable noise parameter entries
        state = 'normal' if self.noise_enabled.get() else 'disabled'
        self.gain_entry.config(state=state)
        self.read_entry.config(state=state)
        self.dark_entry.config(state=state)

    # ----- Object layer helpers -----
    def _add_object(self):
        # Use current image dimensions for sensible defaults
        H, W = int(self.H.get()), int(self.W.get())
        obj = {
            'fluor_index': len(self.objects) % max(1, len(self.fluor_data)),  # Cycle through available fluorophores
            'kind': 'gaussian_blobs',
            'region': {'type': 'rect', 'x0': W//4, 'y0': H//4, 'w': W//2, 'h': H//2},  # Center quarter
            'count': 25,
            'size_px': max(3.0, min(W, H) / 20),  # Scale with image size
            'intensity_min': 0.5,
            'intensity_max': 1.5,
            'spot_sigma': max(1.5, min(W, H) / 40),
        }
        self.objects.append(obj)
        self._refresh_object_list()
        
        # Auto-select the new object
        items = self.obj_tree.get_children()
        if items:
            self.obj_tree.selection_set(items[-1])
            self._on_object_select()
        
        self._log(f"Added object {len(self.objects)}: F{obj['fluor_index']+1}, {obj['kind']}")

    def _remove_object(self):
        sel = self.obj_tree.selection()
        if not sel:
            if self.objects:
                self.objects.pop()
        else:
            idx = self.obj_tree.index(sel[0])
            if 0 <= idx < len(self.objects):
                self.objects.pop(idx)
        self._refresh_object_list()

    def _duplicate_object(self):
        sel = self.obj_tree.selection()
        if not sel:
            return
        idx = self.obj_tree.index(sel[0])
        if 0 <= idx < len(self.objects):
            import copy
            self.objects.append(copy.deepcopy(self.objects[idx]))
            self._refresh_object_list()

    def _refresh_object_list(self):
        # Clear tree
        for i in self.obj_tree.get_children():
            self.obj_tree.delete(i)
        # Populate with simplified format
        for obj in self.objects:
            fluor = obj.get('fluor_index', 0)
            kind = obj.get('kind', '')
            region = obj.get('region', {'type': 'full'})
            rtxt = region.get('type', 'full')
            if rtxt == 'rect':
                rtxt = f"rect({region.get('w',0)}×{region.get('h',0)})"
            elif rtxt == 'circle':
                rtxt = f"circle(r={region.get('r',0):.0f})"
            count = obj.get('count', 0)
            self.obj_tree.insert('', 'end', values=(f"F{fluor+1}", kind, rtxt, count))
        
    def _on_object_select(self, event=None):
        sel = self.obj_tree.selection()
        if not sel:
            return
        idx = self.obj_tree.index(sel[0])
        if not (0 <= idx < len(self.objects)):
            return
        obj = self.objects[idx]
        self.obj_fluor.set(int(obj.get('fluor_index', 0)))
        self.obj_kind.set(str(obj.get('kind', 'gaussian_blobs')))
        self.obj_count.set(int(obj.get('count', 50)))
        self.obj_size.set(float(obj.get('size_px', 6.0)))
        self.obj_i_min.set(float(obj.get('intensity_min', 0.5)))
        self.obj_i_max.set(float(obj.get('intensity_max', 1.5)))
        self.obj_sigma.set(float(obj.get('spot_sigma', 2.0)))
        region = obj.get('region', {'type': 'full'})
        rtype = region.get('type', 'full')
        self.obj_region_type.set(rtype)
        self.obj_x0.set(int(region.get('x0', 0)))
        self.obj_y0.set(int(region.get('y0', 0)))
        self.obj_w.set(int(region.get('w', 64)))
        self.obj_h.set(int(region.get('h', 64)))
        self.obj_cx.set(float(region.get('cx', 64)))
        self.obj_cy.set(float(region.get('cy', 64)))
        self.obj_r.set(float(region.get('r', 40)))
        
        # Update the region UI to show correct parameters
        self._update_region_ui()

    def _apply_object_edits(self):
        sel = self.obj_tree.selection()
        if not sel:
            self._log("No object selected for editing")
            return
        idx = self.obj_tree.index(sel[0])
        if not (0 <= idx < len(self.objects)):
            self._log("Invalid object selection")
            return
        
        try:
            region_type = self.obj_region_type.get()
            region = {'type': region_type}
            if region_type == 'rect':
                region.update({'x0': self.obj_x0.get(), 'y0': self.obj_y0.get(), 'w': self.obj_w.get(), 'h': self.obj_h.get()})
            elif region_type == 'circle':
                region.update({'cx': self.obj_cx.get(), 'cy': self.obj_cy.get(), 'r': self.obj_r.get()})
            
            self.objects[idx] = {
                'fluor_index': int(self.obj_fluor.get()),
                'kind': self.obj_kind.get(),
                'region': region,
                'count': int(self.obj_count.get()),
                'size_px': float(self.obj_size.get()),
                'intensity_min': float(self.obj_i_min.get()),
                'intensity_max': float(self.obj_i_max.get()),
                'spot_sigma': float(self.obj_sigma.get()),
            }
            self._refresh_object_list()
            
            # Re-select the updated item and provide feedback
            items = self.obj_tree.get_children()
            if idx < len(items):
                self.obj_tree.selection_set(items[idx])
            self._log(f"Updated object {idx+1}: F{self.obj_fluor.get()+1}, {self.obj_kind.get()}, {region_type}")
            
        except Exception as e:
            self._log(f"Error updating object: {str(e)}")

    def _on_region_type_change(self, event=None):
        # Update UI to show only relevant parameters
        self._update_region_ui()
        
        # Visual feedback when changing region type
        region_type = self.obj_region_type.get()
        if region_type == "full":
            self._log("Region set to full image")
        elif region_type == "rect":
            self._log("Region set to rectangle - configure x0, y0, width, height")
        elif region_type == "circle":
            self._log("Region set to circle - configure center and radius")
    
    def _update_region_ui(self):
        # Clear existing widgets
        for widget in self.region_params_frame.winfo_children():
            widget.destroy()
            
        region_type = self.obj_region_type.get()
        
        if region_type == "rect":
            # Rectangle parameters only
            rect_frame = ttk.LabelFrame(self.region_params_frame, text="Rectangle Parameters")
            rect_frame.pack(fill='x', pady=(6,0))
            
            for i, (label, var) in enumerate([("x0:", self.obj_x0), ("y0:", self.obj_y0), ("width:", self.obj_w), ("height:", self.obj_h)]):
                ttk.Label(rect_frame, text=label).grid(row=0, column=i*2, sticky='w', padx=(0,2))
                ttk.Entry(rect_frame, textvariable=var, width=6).grid(row=0, column=i*2+1, sticky='w', padx=(0,8))
                
        elif region_type == "circle":
            # Circle parameters only
            circle_frame = ttk.LabelFrame(self.region_params_frame, text="Circle Parameters")
            circle_frame.pack(fill='x', pady=(6,0))
            
            for i, (label, var) in enumerate([("center_x:", self.obj_cx), ("center_y:", self.obj_cy), ("radius:", self.obj_r)]):
                ttk.Label(circle_frame, text=label).grid(row=0, column=i*2, sticky='w', padx=(0,2))
                ttk.Entry(circle_frame, textvariable=var, width=8).grid(row=0, column=i*2+1, sticky='w', padx=(0,8))
                
        else:  # "full"
            # No parameters needed for full image
            ttk.Label(self.region_params_frame, text="Full image - no parameters needed", 
                     foreground="gray", font=('TkDefaultFont', 9, 'italic')).pack(pady=10)

    def _log(self, msg: str) -> None:
        self.output.insert(tk.END, msg + "\n")
        self.output.see(tk.END)

    def on_generate(self) -> None:
        try:
            rng = np.random.default_rng(int(self.seed.get()))
            start = float(self.grid_start.get())
            stop = float(self.grid_stop.get())
            step = float(self.grid_step.get())
            lambdas = np.arange(start, stop + 1e-9, step, dtype=np.float32)

            L = int(self.num_channels.get())
            bw = float(self.bandwidth.get())
            centers = np.linspace(start + 0.5 * bw, stop - 0.5 * bw, L)
            channels = [Channel(name=f"C{idx+1}", center_nm=float(c), bandwidth_nm=bw) for idx, c in enumerate(centers)]

            # Get fluorophores from data
            fluors = []
            for data in self.fluor_data:
                fluor = Fluorophore(
                    name=data['name'],
                    model=data['model'],
                    params=data['params'],
                    brightness=data['brightness']
                )
                fluors.append(fluor)
            K = len(fluors)

            spectral = SpectralSystem(lambdas=lambdas, channels=channels, fluors=fluors)
            M = spectral.build_M()

            H = int(self.H.get())
            W = int(self.W.get())
            px_nm = float(self.pixel_nm.get())
            field = FieldSpec(shape=(H, W), pixel_size_nm=px_nm)
            af = AbundanceField(rng)
            # Build base/global field if enabled
            base_A = None
            if self.include_base_field.get():
                kind = self.spatial_kind.get()
                if kind == "dots":
                    base_A = af.sample(
                        K=K,
                        field=field,
                        kind="dots",
                        density_per_100x100_um2=float(self.density.get()),
                        spot_profile={"kind": "gaussian", "sigma_px": float(self.spot_sigma.get())},
                    )
                elif kind == "uniform":
                    base_A = af.sample(K=K, field=field, kind="uniform")
                else:
                    base_A = af.sample(
                        K=K,
                        field=field,
                        kind=kind,
                        count_per_fluor=int(self.count_per_fluor.get()),
                        size_px=float(self.size_px.get()),
                        intensity_min=float(self.intensity_min.get()),
                        intensity_max=float(self.intensity_max.get()),
                    )

            # If objects exist, build from objects (with optional base)
            if len(self.objects) > 0:
                A = af.build_from_objects(K=K, field=field, objects=self.objects, base=base_A)
            else:
                A = base_A if base_A is not None else af.sample(K=K, field=field, kind="uniform")

            bg = BackgroundModel(rng)
            noise = NoiseModel(rng)
            fwd = ForwardModel(spectral=spectral, field=field, bg=bg, noise=noise, cfg=ForwardConfig())
            B = bg.sample(M.shape[0], H, W, kind="constant", level=0.0)
            
            # Apply noise based on toggle
            if self.noise_enabled.get():
                noise_params = {
                    "gain": float(self.gain.get()), 
                    "read_sigma": float(self.read_sigma.get()), 
                    "dark_rate": float(self.dark_rate.get())
                }
            else:
                # No noise - perfect signal
                noise_params = {
                    "gain": 1.0, 
                    "read_sigma": 0.0, 
                    "dark_rate": 0.0
                }
            
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

            self._log(f"Generated data: L={L}, K={K}, HxW={H}x{W}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_unmix(self) -> None:
        try:
            if self.current_Y is None:
                messagebox.showinfo("Info", "Generate data first.")
                return
            Y = self.current_Y
            M = self.current_M
            A = self.current_A
            B = self.current_B
            H, W = self.current_field.shape

            methods = []
            if self.use_nnls.get():
                methods.append({"method": "nnls"})
            if self.use_lasso.get():
                methods.append({"method": "lasso", "alpha": float(self.lasso_alpha.get())})
            if self.use_nmf.get():
                methods.append({"method": "nmf", "K": M.shape[1], "n_iter": int(self.nmf_iters.get())})
            if self.use_em.get():
                methods.append({"method": "em_poisson", "n_iter": int(self.em_iters.get())})

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

    def _populate_channel_toggles(self, L: int) -> None:
        for w in self.channel_checks_frame.winfo_children():
            w.destroy()
        self.channel_vars = []
        for i in range(L):
            var = tk.BooleanVar(value=True)
            self.channel_vars.append(var)
            ttk.Checkbutton(self.channel_checks_frame, text=f"C{i+1}", variable=var, command=self._render_data_view).grid(row=i//4, column=i%4, sticky="w")

    def _select_all_channels(self) -> None:
        for v in self.channel_vars:
            v.set(True)
        self._render_data_view()

    def _select_no_channels(self) -> None:
        for v in self.channel_vars:
            v.set(False)
        self._render_data_view()

    def _populate_fluor_toggles(self, K: int) -> None:
        for w in self.fluor_checks_frame.winfo_children():
            w.destroy()
        self.fluor_vars = []
        for i in range(K):
            var = tk.BooleanVar(value=True)
            self.fluor_vars.append(var)
            ttk.Checkbutton(self.fluor_checks_frame, text=f"F{i+1}", variable=var, command=self._render_data_view).grid(row=i//4, column=i%4, sticky="w")

    def _select_all_fluors(self) -> None:
        for v in self.fluor_vars:
            v.set(True)
        self._render_data_view()

    def _select_no_fluors(self) -> None:
        for v in self.fluor_vars:
            v.set(False)
        self._render_data_view()

    def _render_data_view(self) -> None:
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
        if self.show_regions.get() and len(self.objects) > 0 and self.current_field is not None:
            H, W = self.current_field.shape
            overlay = np.zeros((H, W, 3), dtype=np.float32)
            # Cycle colors
            colors = [np.array([1.0,0.0,0.0]), np.array([0.0,1.0,0.0]), np.array([0.0,0.0,1.0]), np.array([1.0,1.0,0.0])]
            color_idx = 0
            for obj in self.objects:
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

        # Spectral panel (better organized)
        ax_spec = self.data_figure.add_subplot(2, 2, 3)
        start = float(self.grid_start.get())
        stop = float(self.grid_stop.get())
        ax_spec.set_xlim(start, stop)
        ax_spec.set_ylim(0, 1.1)
        ax_spec.set_xlabel("Wavelength (nm)")
        ax_spec.set_ylabel("Normalized Intensity")
        ax_spec.set_title("Spectral Profiles", fontsize=10)
        
        # Draw channel bands more clearly
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
                
                 # Overlay fluor PDFs and measured total signal
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
            # Sum the measured intensities across all pixels for each channel
            total_per_channel = np.sum(self.current_Y, axis=1)  # Shape: (L,)

            channel_centers = np.array([ch.center_nm for ch in self.current_spectral.channels])
            
            # Interpolate the discrete channel totals across the wavelength grid
            # Then smooth with a small Gaussian for visual continuity
            if len(channel_centers) >= 2:
                # Interpolate
                interp = np.interp(self.current_spectral.lambdas, channel_centers, total_per_channel, 
                                   left=total_per_channel[0], right=total_per_channel[-1])
            else:
                interp = np.full_like(self.current_spectral.lambdas, total_per_channel[0] if len(total_per_channel) > 0 else 0.0)

            # Simple smoothing via moving average over ~5 nm window
            lambdas = self.current_spectral.lambdas
            if len(lambdas) > 3:
                # choose window based on grid step
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

        # GT abundance maps for selected fluorophores
        if self.current_A is not None and len(active_fl) > 0:
            ax_maps = self.data_figure.add_subplot(2, 2, 4)
            n_show = min(len(active_fl), 1)  # Show only one at a time for clarity
            if n_show > 0:
                k = active_fl[0]  # Show first selected
                im = ax_maps.imshow(self.current_A[k].reshape(H, W), cmap="magma")
                ax_maps.set_title(f"Ground Truth: F{k+1} Abundance", fontsize=10)
                ax_maps.axis("off")
                self.data_figure.colorbar(im, ax=ax_maps, fraction=0.046, pad=0.04)
            
        self.data_canvas.draw_idle()

    @staticmethod
    def _wavelength_to_rgb_nm(nm: float) -> tuple[float, float, float]:
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

    def _toggle_settings(self) -> None:
        if self.settings_visible:
            self.settings_container.pack_forget()
            self.toggle_btn.config(text="Show Settings")
            self.settings_visible = False
        else:
            self.settings_container.pack(fill=tk.BOTH, expand=True)
            self.toggle_btn.config(text="Hide Settings")
            self.settings_visible = True


def main() -> None:
    app = PlaygroundGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
