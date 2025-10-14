"""UI components for object layers management."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .manager import ObjectLayersManager


class ObjectLayersUI:
    """Handles UI building for ObjectLayersManager."""
    
    @staticmethod
    def build_main_ui(manager: 'ObjectLayersManager') -> None:
        """Build the complete object layers management UI.
        
        Args:
            manager: ObjectLayersManager instance with all state and callbacks
        """
        # Help text only
        help_frame = ttk.Frame(manager.parent_frame)
        help_frame.pack(fill=tk.X, padx=4, pady=(4,0))
        ttk.Label(help_frame, text="Place specific fluorophores in regions", 
                 foreground="gray", font=('TkDefaultFont', 8)).pack(side=tk.LEFT)

        obj_main = ttk.Frame(manager.parent_frame)
        obj_main.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        obj_main.columnconfigure(0, weight=1)

        # Object list with improved buttons
        list_frame = ttk.Frame(obj_main)
        list_frame.grid(row=0, column=0, sticky="nsew", pady=(0,4))
        list_frame.columnconfigure(0, weight=1)
        
        header_frame = ttk.Frame(list_frame)
        header_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0,4))
        header_frame.columnconfigure(0, weight=1)
        
        # All buttons in a single row
        btns = ttk.Frame(header_frame)
        btns.pack(side=tk.LEFT)
        ttk.Button(btns, text="Add", width=6, command=manager._add_object).pack(side=tk.LEFT, padx=(0,2))
        ttk.Button(btns, text="Remove", width=6, command=manager._remove_object).pack(side=tk.LEFT, padx=(0,2))
        ttk.Button(btns, text="Copy", width=6, command=manager._duplicate_object).pack(side=tk.LEFT, padx=(0,2))
        ttk.Button(btns, text="Quick-Assign", width=12, command=manager._quick_assign_sample).pack(side=tk.LEFT)

        # Simplified object list
        manager.obj_tree = ttk.Treeview(list_frame, columns=("fluor","kind","region","count"), 
                                       show='headings', height=4)
        for col, text, w in [("fluor","Fluor",50),("kind","Kind",80),("region","Region",80),("count","Count",50)]:
            manager.obj_tree.heading(col, text=text, anchor='w')
            manager.obj_tree.column(col, width=w, stretch=True, anchor='w')
        manager.obj_tree.grid(row=1, column=0, sticky='nsew')
        obj_scroll = ttk.Scrollbar(list_frame, orient='vertical', command=manager.obj_tree.yview)
        manager.obj_tree.configure(yscrollcommand=obj_scroll.set)
        obj_scroll.grid(row=1, column=1, sticky='ns')
        manager.obj_tree.bind('<<TreeviewSelect>>', manager._on_object_select)

        # Compact editor with tabs
        editor_notebook = ttk.Notebook(obj_main)
        editor_notebook.grid(row=1, column=0, sticky='ew', pady=(4,0))
        
        ObjectLayersUI.build_properties_tab(manager, editor_notebook)
        ObjectLayersUI.build_region_tab(manager, editor_notebook)
    
    @staticmethod
    def build_properties_tab(manager: 'ObjectLayersManager', notebook: ttk.Notebook) -> None:
        """Build the properties tab for object editing with template support."""
        basic_tab = ttk.Frame(notebook)
        notebook.add(basic_tab, text="Properties")
        
        # Row 0: Composition Mode Selection
        mode_frame = ttk.LabelFrame(basic_tab, text="Composition Mode")
        mode_frame.grid(row=0, column=0, columnspan=4, sticky='ew', pady=(0,8))
        
        ttk.Radiobutton(mode_frame, text="Single Fluorophore", 
                       variable=manager.composition_mode, value="single",
                       command=manager._on_composition_mode_change).pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(mode_frame, text="Multi-Fluorophore Template", 
                       variable=manager.composition_mode, value="template",
                       command=manager._on_composition_mode_change).pack(side=tk.LEFT, padx=4)
        
        # Row 1: Fluorophore/Template Selection (dynamic based on mode)
        manager.fluor_selection_frame = ttk.Frame(basic_tab)
        manager.fluor_selection_frame.grid(row=1, column=0, columnspan=4, sticky='ew', pady=(0,8))
        
        # Single fluorophore selector
        manager.single_fluor_frame = ttk.Frame(manager.fluor_selection_frame)
        ttk.Label(manager.single_fluor_frame, text="Fluorophore:").pack(side=tk.LEFT, padx=(0,4))
        manager.fluor_combo = ttk.Combobox(manager.single_fluor_frame, textvariable=manager.obj_fluor, 
                                          values=manager._get_fluorophore_list(), 
                                          state='readonly', width=15)
        manager.fluor_combo.pack(side=tk.LEFT)
        manager.fluor_combo.bind('<<ComboboxSelected>>', lambda e: manager._auto_save())
        
        # Template selector
        manager.template_frame = ttk.Frame(manager.fluor_selection_frame)
        ttk.Label(manager.template_frame, text="Template:").pack(side=tk.LEFT, padx=(0,4))
        manager.template_combo = ttk.Combobox(manager.template_frame, 
                                             textvariable=manager.composition_template,
                                             values=manager.template_manager.get_template_names(),
                                             state='readonly', width=20)
        manager.template_combo.pack(side=tk.LEFT, padx=(0,4))
        manager.template_combo.bind('<<ComboboxSelected>>', lambda e: manager._auto_save())
        
        ttk.Button(manager.template_frame, text="Manage Templates", 
                  command=manager._open_template_manager, width=15).pack(side=tk.LEFT)
        
        # Row 2: Kind
        ttk.Label(basic_tab, text="Kind:").grid(row=2, column=0, sticky='w', padx=(0,4))
        manager.kind_combo = ttk.Combobox(basic_tab, textvariable=manager.obj_kind, 
                                         values=["circles","boxes","gaussian_blobs","dots"], 
                                         state='readonly', width=15)
        manager.kind_combo.grid(row=2, column=1, columnspan=3, sticky='w')
        manager.kind_combo.bind('<<ComboboxSelected>>', lambda e: (manager._on_kind_change(), manager._auto_save()))

        # Row 3: Count and Lambda (synchronized)
        count_lambda_frame = ttk.LabelFrame(basic_tab, text="Object Density")
        count_lambda_frame.grid(row=3, column=0, columnspan=4, sticky='ew', pady=(6,0))
        
        # Left column: Count
        count_col = ttk.Frame(count_lambda_frame)
        count_col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4, pady=4)
        
        ttk.Label(count_col, text="Count:").pack(anchor='w')
        count_entry = ttk.Entry(count_col, textvariable=manager.obj_count, width=10)
        count_entry.pack(anchor='w')
        count_entry.bind('<KeyRelease>', lambda e: manager._on_count_changed())
        
        manager.count_display_label = ttk.Label(count_col, text="→ λ = 0.000000 obj/px²", 
                                               font=('TkDefaultFont', 7), foreground='gray')
        manager.count_display_label.pack(anchor='w', pady=(2,0))
        
        # Right column: Lambda
        lambda_col = ttk.Frame(count_lambda_frame)
        lambda_col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4, pady=4)
        
        ttk.Label(lambda_col, text="Spatial intensity (λ):").pack(anchor='w')
        lambda_entry = ttk.Entry(lambda_col, textvariable=manager.obj_lambda, width=10)
        lambda_entry.pack(anchor='w')
        lambda_entry.bind('<KeyRelease>', lambda e: manager._on_lambda_changed())
        
        manager.lambda_display_label = ttk.Label(lambda_col, text="→ n ≈ 0 objects", 
                                                font=('TkDefaultFont', 7), foreground='gray')
        manager.lambda_display_label.pack(anchor='w', pady=(2,0))
        
        # Info label
        ttk.Label(count_lambda_frame, text="ℹ️ Count and spatial intensity are linked: λ = n / area", 
                  font=('TkDefaultFont', 7, 'italic'), foreground='#666').pack(pady=(0,4))
        
        # Row 4: Size
        manager.size_label = ttk.Label(basic_tab, text="Size (px):")
        manager.size_label.grid(row=4, column=0, sticky='w', padx=(0,4), pady=(6,0))
        manager.size_entry = ttk.Entry(basic_tab, textvariable=manager.obj_size, width=10)
        manager.size_entry.grid(row=4, column=1, sticky='w', pady=(6,0))
        manager.size_entry.bind('<KeyRelease>', lambda e: manager._auto_save())

        # Row 5: Intensity range
        ttk.Label(basic_tab, text="Spectral intensity (brightness):").grid(row=5, column=0, sticky='w', pady=(6,0))
        intensity_frame = ttk.Frame(basic_tab)
        intensity_frame.grid(row=5, column=1, columnspan=3, sticky='w', pady=(6,0))
        i_min_entry = ttk.Entry(intensity_frame, textvariable=manager.obj_i_min, width=6)
        i_min_entry.pack(side=tk.LEFT)
        i_min_entry.bind('<KeyRelease>', lambda e: manager._auto_save())
        ttk.Label(intensity_frame, text="to").pack(side=tk.LEFT, padx=2)
        i_max_entry = ttk.Entry(intensity_frame, textvariable=manager.obj_i_max, width=6)
        i_max_entry.pack(side=tk.LEFT)
        i_max_entry.bind('<KeyRelease>', lambda e: manager._auto_save())
        
        # Row 6: Spot sigma (for Gaussian blobs and dots only)
        manager.sigma_label = ttk.Label(basic_tab, text="Spot σ (px):")
        manager.sigma_label.grid(row=6, column=0, sticky='w', pady=(6,0))
        manager.sigma_entry = ttk.Entry(basic_tab, textvariable=manager.obj_sigma, width=10)
        manager.sigma_entry.grid(row=6, column=1, columnspan=3, sticky='w', pady=(6,0))
        manager.sigma_entry.bind('<KeyRelease>', lambda e: manager._auto_save())
        
        # Row 7: Info text about radius derivation
        ttk.Label(
            basic_tab,
            text="ℹ️ Radius derived from object type: circles/boxes=size_px, gaussian=2×σ",
            font=('TkDefaultFont', 7, 'italic'),
            foreground='gray'
        ).grid(row=7, column=0, columnspan=4, sticky='w', pady=(8,0))
        
        # Initialize mode UI
        manager._on_composition_mode_change()

    @staticmethod
    def build_region_tab(manager: 'ObjectLayersManager', notebook: ttk.Notebook) -> None:
        """Build the region tab for object editing."""
        region_tab = ttk.Frame(notebook)
        notebook.add(region_tab, text="Region")
        
        ttk.Label(region_tab, text="Type:").grid(row=0, column=0, sticky='w')
        region_combo = ttk.Combobox(region_tab, textvariable=manager.obj_region_type, 
                                    values=["full","rect","circle"], state='readonly', width=12)
        region_combo.grid(row=0, column=1, sticky='w', padx=(0,8))
        region_combo.bind('<<ComboboxSelected>>', lambda e: (manager._on_region_type_change(), manager._auto_save()))

        # Container for dynamic region parameters
        manager.region_params_frame = ttk.Frame(region_tab)
        manager.region_params_frame.grid(row=1, column=0, columnspan=4, sticky='ew', pady=(6,0))
        
        # Initialize with full region (no parameters)
        ObjectLayersUI.update_region_ui(manager)
    
    @staticmethod
    def update_region_ui(manager: 'ObjectLayersManager') -> None:
        """Update region UI to show only relevant parameters."""
        # Clear existing widgets
        for widget in manager.region_params_frame.winfo_children():
            widget.destroy()
            
        region_type = manager.obj_region_type.get()
        
        if region_type == "rect":
            # Rectangle parameters only
            rect_frame = ttk.LabelFrame(manager.region_params_frame, text="Rectangle Parameters")
            rect_frame.pack(fill='x', pady=(6,0))
            
            for i, (label, var) in enumerate([("x0:", manager.obj_x0), ("y0:", manager.obj_y0), 
                                             ("width:", manager.obj_w), ("height:", manager.obj_h)]):
                ttk.Label(rect_frame, text=label).grid(row=0, column=i*2, sticky='w', padx=(0,2))
                entry = ttk.Entry(rect_frame, textvariable=var, width=6)
                entry.grid(row=0, column=i*2+1, sticky='w', padx=(0,8))
                entry.bind('<KeyRelease>', lambda e: manager._auto_save())
                
        elif region_type == "circle":
            # Circle parameters only
            circle_frame = ttk.LabelFrame(manager.region_params_frame, text="Circle Parameters")
            circle_frame.pack(fill='x', pady=(6,0))
            
            for i, (label, var) in enumerate([("center_x:", manager.obj_cx), ("center_y:", manager.obj_cy), 
                                             ("radius:", manager.obj_r)]):
                ttk.Label(circle_frame, text=label).grid(row=0, column=i*2, sticky='w', padx=(0,2))
                entry = ttk.Entry(circle_frame, textvariable=var, width=8)
                entry.grid(row=0, column=i*2+1, sticky='w', padx=(0,8))
                entry.bind('<KeyRelease>', lambda e: manager._auto_save())
                
        else:  # "full"
            # No parameters needed for full image
            ttk.Label(manager.region_params_frame, text="Full image - no parameters needed", 
                     foreground="gray", font=('TkDefaultFont', 9, 'italic')).pack(pady=10)

