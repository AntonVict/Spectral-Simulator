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
        # Global composition settings
        settings_frame = ttk.LabelFrame(manager.parent_frame, text="Multi-Fluorophore Settings")
        settings_frame.pack(fill=tk.X, padx=4, pady=4)
        
        settings_grid = ttk.Frame(settings_frame)
        settings_grid.pack(fill=tk.X, padx=4, pady=4)
        
        # Row 0: Object Mode
        ttk.Label(settings_grid, text="Object Mode:").grid(row=0, column=0, sticky='w', padx=2, pady=2)
        mode_frame = ttk.Frame(settings_grid)
        mode_frame.grid(row=0, column=1, columnspan=2, sticky='w', padx=2, pady=2)
        ttk.Radiobutton(mode_frame, text="Single Fluorophore", 
                        variable=manager.object_mode, value="single").pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(mode_frame, text="Multi-Fluorophore", 
                        variable=manager.object_mode, value="multi").pack(side=tk.LEFT, padx=4)
        
        # Row 1: Fluorophore Configuration
        ttk.Label(settings_grid, text="Fluorophore Roles:").grid(row=1, column=0, sticky='w', padx=2, pady=2)
        ttk.Button(settings_grid, text="Configure...", 
                   command=manager._open_fluorophore_config).grid(row=1, column=1, sticky='w', padx=2, pady=2)
        
        # Add a label showing current config
        manager.fluor_config_label = ttk.Label(settings_grid, text="F1,F2 (ratio) | F3+ (binary)", 
                                                foreground='gray', font=('TkDefaultFont', 8))
        manager.fluor_config_label.grid(row=1, column=2, sticky='w', padx=2, pady=2)
        
        # Row 2: Dirichlet option
        ttk.Checkbutton(settings_grid, text="Use Dirichlet distribution", 
                        variable=manager.use_dirichlet).grid(row=2, column=0, columnspan=3, sticky='w', padx=2, pady=2)

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
        
        # Two quick-assign buttons
        ttk.Button(btns, text="Quick-Assign (Single)", width=18, 
                  command=manager._quick_assign_single).pack(side=tk.LEFT, padx=(0,2))
        ttk.Button(btns, text="Quick-Assign (Multi)", width=17, 
                  command=manager._quick_assign_multi).pack(side=tk.LEFT)

        # Object list with new columns: Name | Composition | Count | Radius
        manager.obj_tree = ttk.Treeview(list_frame, columns=("name", "composition", "count", "radius"), 
                                       show='headings', height=6)
        columns_config = [
            ("name", "Name", 60),
            ("composition", "Composition", 120),
            ("count", "Count", 50),
            ("radius", "Radius", 60)
        ]
        for col, text, width in columns_config:
            manager.obj_tree.heading(col, text=text, anchor='w')
            manager.obj_tree.column(col, width=width, stretch=True, anchor='w')
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
        """Build the properties tab with composition editor."""
        basic_tab = ttk.Frame(notebook)
        notebook.add(basic_tab, text="Properties")
        
        row = 0
        
        # Object name (editable)
        ttk.Label(basic_tab, text="Name:").grid(row=row, column=0, sticky='w', pady=4, padx=(0,4))
        ttk.Entry(basic_tab, textvariable=manager.obj_name, width=15).grid(row=row, column=1, sticky='w', pady=4)
        manager.obj_name.trace('w', lambda *args: manager.object_editor.auto_save())
        row += 1
        
        # Per-object mode selector
        ttk.Label(basic_tab, text="Mode:").grid(row=row, column=0, sticky='w', pady=4, padx=(0,4))
        mode_frame = ttk.Frame(basic_tab)
        mode_frame.grid(row=row, column=1, columnspan=2, sticky='w', pady=4)
        ttk.Radiobutton(mode_frame, text="Single", variable=manager.obj_mode, 
                       value="single", command=lambda: manager._on_mode_change()).pack(side=tk.LEFT, padx=(0,4))
        ttk.Radiobutton(mode_frame, text="Multi", variable=manager.obj_mode, 
                       value="multi", command=lambda: manager._on_mode_change()).pack(side=tk.LEFT)
        row += 1
        
        # Object kind
        ttk.Label(basic_tab, text="Kind:").grid(row=row, column=0, sticky='w', pady=4, padx=(0,4))
        manager.kind_combo = ttk.Combobox(basic_tab, textvariable=manager.obj_kind, 
                                          values=['gaussian_blobs', 'dots', 'circles', 'boxes'], 
                                          state='readonly', width=15)
        manager.kind_combo.grid(row=row, column=1, sticky='w', pady=4)
        manager.kind_combo.bind('<<ComboboxSelected>>', lambda e: manager.object_editor.auto_save())
        row += 1
        
        # Composition editor container (will be dynamically populated)
        manager.comp_editor_frame = ttk.LabelFrame(basic_tab, text="Composition")
        manager.comp_editor_frame.grid(row=row, column=0, columnspan=4, sticky='ew', pady=8)
        row += 1
        
        # Count
        ttk.Label(basic_tab, text="Count:").grid(row=row, column=0, sticky='w', pady=4, padx=(0,4))
        ttk.Spinbox(basic_tab, from_=1, to=5000, textvariable=manager.obj_count, 
                   width=15).grid(row=row, column=1, sticky='w', pady=4)
        manager.obj_count.trace('w', lambda *args: manager.object_editor.auto_save())
        row += 1
        
        # Radius (not sigma)
        ttk.Label(basic_tab, text="Radius:").grid(row=row, column=0, sticky='w', pady=4, padx=(0,4))
        ttk.Entry(basic_tab, textvariable=manager.obj_radius, width=15).grid(row=row, column=1, sticky='w', pady=4)
        ttk.Label(basic_tab, text="px (effective)").grid(row=row, column=2, sticky='w', pady=4, padx=(4,0))
        # Sync radius ↔ sigma: sigma = radius / 2.0
        manager.obj_radius.trace('w', lambda *args: manager.object_editor.sync_radius_to_sigma())
        row += 1
        
        # Intensity range
        ttk.Label(basic_tab, text="Intensity:").grid(row=row, column=0, sticky='w', pady=4, padx=(0,4))
        int_frame = ttk.Frame(basic_tab)
        int_frame.grid(row=row, column=1, columnspan=2, sticky='w', pady=4)
        ttk.Entry(int_frame, textvariable=manager.obj_i_min, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(int_frame, text="to").pack(side=tk.LEFT, padx=2)
        ttk.Entry(int_frame, textvariable=manager.obj_i_max, width=6).pack(side=tk.LEFT, padx=2)
        manager.obj_i_min.trace('w', lambda *args: manager.object_editor.auto_save())
        manager.obj_i_max.trace('w', lambda *args: manager.object_editor.auto_save())
        row += 1

    @staticmethod
    def build_region_tab(manager: 'ObjectLayersManager', notebook: ttk.Notebook) -> None:
        """Build the region tab for object editing."""
        region_tab = ttk.Frame(notebook)
        notebook.add(region_tab, text="Region")
        
        ttk.Label(region_tab, text="Type:").grid(row=0, column=0, sticky='w')
        region_combo = ttk.Combobox(region_tab, textvariable=manager.obj_region_type, 
                                    values=["full","rect","circle"], state='readonly', width=12)
        region_combo.grid(row=0, column=1, sticky='w', padx=(0,8))
        region_combo.bind('<<ComboboxSelected>>', lambda e: (manager.object_editor.on_region_type_change(), manager.object_editor.auto_save()))

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
                entry.bind('<KeyRelease>', lambda e: manager.object_editor.auto_save())
                
        elif region_type == "circle":
            # Circle parameters only
            circle_frame = ttk.LabelFrame(manager.region_params_frame, text="Circle Parameters")
            circle_frame.pack(fill='x', pady=(6,0))
            
            for i, (label, var) in enumerate([("center_x:", manager.obj_cx), ("center_y:", manager.obj_cy), 
                                             ("radius:", manager.obj_r)]):
                ttk.Label(circle_frame, text=label).grid(row=0, column=i*2, sticky='w', padx=(0,2))
                entry = ttk.Entry(circle_frame, textvariable=var, width=8)
                entry.grid(row=0, column=i*2+1, sticky='w', padx=(0,8))
                entry.bind('<KeyRelease>', lambda e: manager.object_editor.auto_save())
                
        else:  # "full"
            # No parameters needed for full image
            ttk.Label(manager.region_params_frame, text="Full image - no parameters needed", 
                     foreground="gray", font=('TkDefaultFont', 9, 'italic')).pack(pady=10)

