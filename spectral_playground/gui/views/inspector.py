"""Object Inspector View - detailed information about individual generated objects."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Dict, Any, Optional, Callable
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ObjectInspectorView(ttk.Frame):
    """View for inspecting individual objects in the generated dataset."""
    
    def __init__(
        self,
        parent: tk.Widget,
        get_data_callback: Callable,
        get_fluorophore_names_callback: Callable
    ):
        super().__init__(parent)
        self.get_data = get_data_callback
        self.get_fluorophore_names = get_fluorophore_names_callback
        
        # Use pandas DataFrames for efficient filtering
        self.df: pd.DataFrame = pd.DataFrame()  # Full dataset
        self.filtered_df: pd.DataFrame = pd.DataFrame()  # After filters applied
        self.selected_objects: List[int] = []  # List of object IDs
        
        # Selection management
        self.selection_locked = False  # Prevents accidental clearing of pre-selected objects
        
        # Filter state
        self.filter_fluor = tk.StringVar(value="All")
        self.filter_type = tk.StringVar(value="All")
        self.filter_object = tk.StringVar(value="All")  # Filter by object name
        self.filter_binary = tk.StringVar(value="All")  # Filter by binary fluorophore
        self.filter_overlap = tk.StringVar(value="All")  # NEW: Filter by overlap status
        self.filter_radius_min = tk.DoubleVar(value=0.0)  # NEW: Min radius
        self.filter_radius_max = tk.DoubleVar(value=100.0)  # NEW: Max radius
        self.filter_intensity_min = tk.DoubleVar(value=0.0)
        self.filter_intensity_max = tk.DoubleVar(value=10.0)
        self.search_var = tk.StringVar()
        self.search_var.trace('w', lambda *args: self._apply_filters())
        self.show_selected_only = tk.BooleanVar(value=False)
        self.lock_selection = tk.BooleanVar(value=False)
        
        self._build_ui()
        
    def _build_ui(self):
        """Build the inspector UI."""
        # Main layout: toolbar | object list | details
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        
        # Toolbar
        toolbar = ttk.Frame(self)
        toolbar.grid(row=0, column=0, sticky='ew', padx=4, pady=4)
        self._build_toolbar(toolbar)
        
        # Main content area
        content = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        content.grid(row=1, column=0, sticky='nsew', padx=4, pady=4)
        
        # Left side: Object list with filters
        left_panel = ttk.Frame(content)
        content.add(left_panel, weight=2)
        self._build_object_list(left_panel)
        
        # Right side: Object details
        right_panel = ttk.Frame(content)
        content.add(right_panel, weight=1)
        self._build_details_panel(right_panel)
    
    def _build_toolbar(self, parent: ttk.Frame):
        """Build toolbar with selection tools and stats."""
        ttk.Label(parent, text="Object Inspector", font=('TkDefaultFont', 10, 'bold')).pack(side=tk.LEFT, padx=4)
        
        # Stats
        self.stats_label = ttk.Label(parent, text="No objects", foreground="gray")
        self.stats_label.pack(side=tk.LEFT, padx=20)
        
        # Refresh button
        ttk.Button(parent, text="Refresh", command=self.refresh_objects, width=10).pack(side=tk.RIGHT, padx=2)
        
        # Export button
        ttk.Button(parent, text="Export CSV", command=self._export_csv, width=10).pack(side=tk.RIGHT, padx=2)
        
        # Clear selection
        ttk.Button(parent, text="Clear Selection", command=self._clear_selection, width=12).pack(side=tk.RIGHT, padx=2)
    
    def _build_object_list(self, parent: ttk.Frame):
        """Build object list with filters."""
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        
        # Filter panel
        filter_frame = ttk.LabelFrame(parent, text="Filters")
        filter_frame.grid(row=0, column=0, sticky='ew', padx=4, pady=4)
        self._build_filters(filter_frame)
        
        # Object table
        table_frame = ttk.Frame(parent)
        table_frame.grid(row=1, column=0, sticky='nsew', padx=4, pady=4)
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        
        # Treeview with multiple columns (updated for multi-fluorophore objects)
        columns = ('id', 'object_name', 'composition', 'type', 'position', 'radius')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', selectmode='extended')
        
        # Configure columns
        self.tree.heading('id', text='ID', command=lambda: self._sort_by('id'))
        self.tree.heading('object_name', text='Object', command=lambda: self._sort_by('object_name'))
        self.tree.heading('composition', text='Composition', command=lambda: self._sort_by('composition'))
        self.tree.heading('type', text='Type', command=lambda: self._sort_by('type'))
        self.tree.heading('position', text='Position (y,x)', command=lambda: self._sort_by('position'))
        self.tree.heading('radius', text='Radius', command=lambda: self._sort_by('radius'))
        
        self.tree.column('id', width=50, stretch=False)
        self.tree.column('object_name', width=80, stretch=False)
        self.tree.column('composition', width=120, stretch=True)
        self.tree.column('type', width=80, stretch=False)
        self.tree.column('position', width=100, stretch=False)
        self.tree.column('radius', width=70, stretch=False)
        
        self.tree.grid(row=0, column=0, sticky='nsew')
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        # Bind selection event
        self.tree.bind('<<TreeviewSelect>>', self._on_object_select)
        
    def _build_filters(self, parent: ttk.Frame):
        """Build filter controls."""
        # Row 0: Search and checkboxes
        ttk.Label(parent, text="Search:").grid(row=0, column=0, sticky='w', padx=2, pady=2)
        ttk.Entry(parent, textvariable=self.search_var, width=15).grid(row=0, column=1, sticky='ew', padx=2, pady=2)
        
        # Show Selected Only checkbox
        ttk.Checkbutton(parent, text="Selected Only", 
                       variable=self.show_selected_only,
                       command=self._apply_filters).grid(row=0, column=2, sticky='w', padx=(10,2), pady=2)
        
        # Lock Selection checkbox
        ttk.Checkbutton(parent, text="Lock Selection", 
                       variable=self.lock_selection).grid(row=0, column=3, sticky='w', padx=2, pady=2)
        
        # Row 0: Fluorophore filter
        ttk.Label(parent, text="Fluorophore:").grid(row=0, column=4, sticky='w', padx=(10,2), pady=2)
        self.fluor_combo = ttk.Combobox(parent, textvariable=self.filter_fluor, state='readonly', width=12)
        self.fluor_combo.grid(row=0, column=5, sticky='ew', padx=2, pady=2)
        self.fluor_combo.bind('<<ComboboxSelected>>', lambda e: self._apply_filters())
        
        # Row 1: Type, Object name, Binary filters
        ttk.Label(parent, text="Type:").grid(row=1, column=0, sticky='w', padx=2, pady=2)
        type_combo = ttk.Combobox(parent, textvariable=self.filter_type, 
                                  values=["All", "dots", "gaussian_blobs", "circles", "boxes"],
                                  state='readonly', width=12)
        type_combo.grid(row=1, column=1, sticky='ew', padx=2, pady=2)
        type_combo.bind('<<ComboboxSelected>>', lambda e: self._apply_filters())
        
        ttk.Label(parent, text="Object:").grid(row=1, column=2, sticky='w', padx=(10,2), pady=2)
        self.object_combo = ttk.Combobox(parent, textvariable=self.filter_object, state='readonly', width=10)
        self.object_combo.grid(row=1, column=3, sticky='ew', padx=2, pady=2)
        self.object_combo.bind('<<ComboboxSelected>>', lambda e: self._apply_filters())
        
        ttk.Label(parent, text="Binary:").grid(row=1, column=4, sticky='w', padx=(10,2), pady=2)
        self.binary_combo = ttk.Combobox(parent, textvariable=self.filter_binary, state='readonly', width=10)
        self.binary_combo.grid(row=1, column=5, sticky='ew', padx=2, pady=2)
        self.binary_combo.bind('<<ComboboxSelected>>', lambda e: self._apply_filters())
        
        # Row 2: Radius, Overlap, Apply button
        ttk.Label(parent, text="Radius:").grid(row=2, column=0, sticky='w', padx=2, pady=2)
        radius_frame = ttk.Frame(parent)
        radius_frame.grid(row=2, column=1, sticky='ew', padx=2, pady=2)
        ttk.Entry(radius_frame, textvariable=self.filter_radius_min, width=6).pack(side=tk.LEFT)
        ttk.Label(radius_frame, text="to").pack(side=tk.LEFT, padx=2)
        ttk.Entry(radius_frame, textvariable=self.filter_radius_max, width=6).pack(side=tk.LEFT)
        
        ttk.Label(parent, text="Overlap:").grid(row=2, column=2, sticky='w', padx=(10,2), pady=2)
        self.overlap_combo = ttk.Combobox(parent, textvariable=self.filter_overlap, state='readonly', width=10)
        self.overlap_combo['values'] = ["All", "Isolated (0)", "Has overlaps (1+)", "1 neighbor", "2-4 neighbors", "5+ neighbors"]
        self.overlap_combo.grid(row=2, column=3, sticky='ew', padx=2, pady=2)
        self.overlap_combo.bind('<<ComboboxSelected>>', lambda e: self._apply_filters())
        
        # Row 3: Intensity range and Apply button
        ttk.Label(parent, text="Intensity:").grid(row=3, column=0, sticky='w', padx=2, pady=2)
        intensity_frame = ttk.Frame(parent)
        intensity_frame.grid(row=3, column=1, sticky='ew', padx=2, pady=2)
        ttk.Entry(intensity_frame, textvariable=self.filter_intensity_min, width=6).pack(side=tk.LEFT)
        ttk.Label(intensity_frame, text="to").pack(side=tk.LEFT, padx=2)
        ttk.Entry(intensity_frame, textvariable=self.filter_intensity_max, width=6).pack(side=tk.LEFT)
        
        # Apply button
        ttk.Button(parent, text="Apply Filters", command=self._apply_filters, width=12).grid(row=3, column=3, padx=4, pady=2)
        
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(3, weight=1)
        parent.columnconfigure(5, weight=1)
    
    def _build_details_panel(self, parent: ttk.Frame):
        """Build details panel for selected object(s)."""
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        
        # Header
        header = ttk.Frame(parent)
        header.grid(row=0, column=0, sticky='ew', padx=4, pady=4)
        self.details_label = ttk.Label(header, text="Select an object to view details", 
                                       font=('TkDefaultFont', 9, 'bold'))
        self.details_label.pack(side=tk.LEFT)
        
        # Scrollable details
        details_frame = ttk.Frame(parent)
        details_frame.grid(row=1, column=0, sticky='nsew', padx=4, pady=4)
        details_frame.columnconfigure(0, weight=1)
        details_frame.rowconfigure(0, weight=1)
        
        canvas = tk.Canvas(details_frame, bg='white')
        scrollbar = ttk.Scrollbar(details_frame, orient='vertical', command=canvas.yview)
        self.details_content = ttk.Frame(canvas)
        
        self.details_content.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=self.details_content, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        # Initial empty state
        ttk.Label(self.details_content, text="No object selected", foreground="gray").pack(pady=20)
    
    def refresh_objects(self):
        """Refresh object list from current dataset."""
        data = self.get_data()
        if not data or not data.has_data:
            self.df = pd.DataFrame()
            self.filtered_df = pd.DataFrame()
            self._update_tree()
            self._update_stats()
            return
        
        # Get objects from metadata and convert to DataFrame
        objects_list = data.metadata.get('objects', [])
        if not objects_list:
            self.df = pd.DataFrame()
            self.filtered_df = pd.DataFrame()
            self._update_tree()
            self._update_stats()
            return
        
        # Convert to DataFrame - pandas handles list of dicts naturally
        self.df = pd.DataFrame(objects_list)
        
        # Compute neighbor counts if geometric scene is available
        if data.has_geometric_data:
            scene = data.geometric_scene
            neighbor_counts = []
            for obj in scene.objects:
                n_neighbors = len(scene.get_neighbors(obj.id))
                neighbor_counts.append(n_neighbors)
            
            # Add neighbor_count column to DataFrame
            self.df['neighbor_count'] = neighbor_counts
        else:
            self.df['neighbor_count'] = 0  # No geometric data
        
        # Auto-detect and set filter ranges
        if 'radius' in self.df.columns and len(self.df) > 0:
            min_r = float(self.df['radius'].min())
            max_r = float(self.df['radius'].max())
            self.filter_radius_min.set(min_r)
            self.filter_radius_max.set(max_r * 1.1)  # Add 10% headroom
        
        if 'base_intensity' in self.df.columns and len(self.df) > 0:
            min_int = float(self.df['base_intensity'].min())
            max_int = float(self.df['base_intensity'].max())
            self.filter_intensity_min.set(min_int)
            self.filter_intensity_max.set(max_int * 1.1)  # Add 10% headroom
        
        # Update filter options
        fluor_names = self.get_fluorophore_names()
        self.fluor_combo.config(values=["All"] + fluor_names)
        
        # Update object name filter options
        if 'object_name' in self.df.columns:
            object_names = sorted(self.df['object_name'].unique())
            self.object_combo.config(values=["All"] + list(object_names))
        
        # Update binary fluorophore filter options
        if 'binary_fluor' in self.df.columns:
            binary_indices = self.df['binary_fluor'].dropna().unique()
            binary_names = [fluor_names[int(idx)] for idx in binary_indices if idx < len(fluor_names)]
            self.binary_combo.config(values=["All"] + sorted(binary_names))
        
        # Apply current filters
        self._apply_filters()
        
    def _apply_filters(self):
        """Apply current filters to object list using pandas."""
        if self.df.empty:
            self.filtered_df = pd.DataFrame()
            self._update_tree()
            self._update_stats()
            return
        
        # Start with full dataset
        filtered = self.df.copy()
        
        # Filter 1: Show selected only (if enabled)
        if self.show_selected_only.get() and self.selected_objects:
            filtered = filtered[filtered['id'].isin(self.selected_objects)]
        
        # Filter 2: Type filter
        type_filter = self.filter_type.get()
        if type_filter != "All":
            filtered = filtered[filtered['type'] == type_filter]
        
        # Filter 3: Intensity range
        intensity_min = self.filter_intensity_min.get()
        intensity_max = self.filter_intensity_max.get()
        filtered = filtered[
            (filtered['base_intensity'] >= intensity_min) &
            (filtered['base_intensity'] <= intensity_max)
        ]
        
        # Filter 4: Fluorophore filter
        fluor_filter = self.filter_fluor.get()
        if fluor_filter != "All":
            fluor_names = self.get_fluorophore_names()
            try:
                fluor_idx = fluor_names.index(fluor_filter)
                # Filter by checking if fluorophore is in composition
                def has_fluorophore(composition):
                    return any(c['fluor_index'] == fluor_idx for c in composition)
                
                filtered = filtered[filtered['composition'].apply(has_fluorophore)]
            except (ValueError, KeyError):
                pass  # Invalid fluorophore, skip filter
        
        # Filter 5: Object name filter
        object_filter = self.filter_object.get()
        if object_filter != "All" and 'object_name' in filtered.columns:
            filtered = filtered[filtered['object_name'] == object_filter]
        
        # Filter 6: Binary fluorophore filter
        binary_filter = self.filter_binary.get()
        if binary_filter != "All" and 'binary_fluor' in filtered.columns:
            fluor_names = self.get_fluorophore_names()
            try:
                binary_idx = fluor_names.index(binary_filter)
                filtered = filtered[filtered['binary_fluor'] == binary_idx]
            except (ValueError, KeyError):
                pass  # Invalid fluorophore, skip filter
        
        # Filter 7: Radius range
        if 'radius' in filtered.columns:
            radius_min = self.filter_radius_min.get()
            radius_max = self.filter_radius_max.get()
            filtered = filtered[
                (filtered['radius'] >= radius_min) &
                (filtered['radius'] <= radius_max)
            ]
        
        # Filter 8: Overlap status
        overlap_filter = self.filter_overlap.get()
        if overlap_filter != "All" and 'neighbor_count' in filtered.columns:
            if overlap_filter == "Isolated (0)":
                filtered = filtered[filtered['neighbor_count'] == 0]
            elif overlap_filter == "Has overlaps (1+)":
                filtered = filtered[filtered['neighbor_count'] >= 1]
            elif overlap_filter == "1 neighbor":
                filtered = filtered[filtered['neighbor_count'] == 1]
            elif overlap_filter == "2-4 neighbors":
                filtered = filtered[(filtered['neighbor_count'] >= 2) & (filtered['neighbor_count'] <= 4)]
            elif overlap_filter == "5+ neighbors":
                filtered = filtered[filtered['neighbor_count'] >= 5]
        
        # Filter 9: Text search
        search_text = self.search_var.get().lower()
        if search_text:
            # Search in ID, object name, and position columns
            def matches_search(row):
                id_str = str(row['id']).lower()
                pos_str = str(row['position']).lower()
                obj_name = str(row.get('object_name', '')).lower()
                return search_text in id_str or search_text in pos_str or search_text in obj_name
            
            filtered = filtered[filtered.apply(matches_search, axis=1)]
        
        self.filtered_df = filtered
        self._update_tree()
        self._update_stats()
    
    def _update_tree(self):
        """Update tree view with filtered objects."""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        if self.filtered_df.empty:
            return
        
        # Get fluorophore names
        fluor_names = self.get_fluorophore_names()
        
        # Add filtered objects to tree
        for idx, row in self.filtered_df.iterrows():
            obj_id = row['id']
            
            # Object name
            obj_name = row.get('object_name', 'Unknown')
            
            # Get composition string (binary first, then continuous)
            from spectral_playground.gui.objects.composition import CompositionGenerator
            comp_display = CompositionGenerator.composition_to_display(
                row['composition'],
                row.get('binary_fluor')
            )
            
            obj_type = row['type']
            pos = row['position']
            pos_str = f"({pos[1]:.1f}, {pos[0]:.1f})"
            radius = row.get('radius', row.get('spot_sigma', 0) * 2.0)
            
            self.tree.insert('', 'end', iid=str(obj_id), values=(
                obj_id, 
                obj_name, 
                comp_display, 
                obj_type, 
                pos_str, 
                f"{radius:.1f}px"
            ))
    
    def _update_stats(self):
        """Update statistics label."""
        total = len(self.df)
        filtered = len(self.filtered_df)
        selected = len(self.selected_objects)
        
        if total == 0:
            self.stats_label.config(text="No objects")
        elif filtered == total:
            self.stats_label.config(text=f"{total} objects | {selected} selected")
        else:
            self.stats_label.config(text=f"{filtered} of {total} objects | {selected} selected")
    
    def _on_object_select(self, event=None):
        """Handle object selection in tree."""
        # Check if selection is locked
        if self.lock_selection.get() and self.selection_locked:
            # Restore previous selection
            self.tree.selection_remove(self.tree.selection())
            for obj_id in self.selected_objects:
                try:
                    self.tree.selection_add(str(obj_id))
                except:
                    pass  # Object not in filtered view
            return
        
        # Unlock if user explicitly clears selection
        if not self.lock_selection.get():
            self.selection_locked = False
        
        selection = self.tree.selection()
        self.selected_objects = [int(item) for item in selection]
        self._update_stats()
        self._show_object_details()
    
    def _show_object_details(self):
        """Show details for selected object(s)."""
        # Clear existing details
        for widget in self.details_content.winfo_children():
            widget.destroy()
        
        if not self.selected_objects:
            ttk.Label(self.details_content, text="No object selected", foreground="gray").pack(pady=20)
            self.details_label.config(text="Select an object to view details")
            return
        
        if len(self.selected_objects) == 1:
            obj_id = self.selected_objects[0]
            # Get object from DataFrame
            obj_row = self.df[self.df['id'] == obj_id]
            if not obj_row.empty:
                obj = obj_row.iloc[0].to_dict()
                self._show_single_object_details(obj)
        else:
            self._show_multiple_objects_summary()
    
    def _show_single_object_details(self, obj: Dict[str, Any]):
        """Show detailed information for a single object."""
        self.details_label.config(text=f"Object #{obj['id']}")
        
        fluor_names = self.get_fluorophore_names()
        
        # Zoomed composite view
        self._add_zoomed_composite_view(obj)
        
        # Basic properties
        props_frame = ttk.LabelFrame(self.details_content, text="Properties")
        props_frame.pack(fill=tk.X, padx=4, pady=4)
        
        ttk.Label(props_frame, text=f"ID: {obj['id']}").pack(anchor='w', padx=4, pady=2)
        ttk.Label(props_frame, text=f"Type: {obj.get('type', 'unknown')}").pack(anchor='w', padx=4, pady=2)
        pos = obj.get('position', (0, 0))
        # Display as (x, y) for user clarity (stored order is (y, x))
        ttk.Label(props_frame, text=f"Position: ({pos[1]:.2f}, {pos[0]:.2f}) px").pack(anchor='w', padx=4, pady=2)
        ttk.Label(props_frame, text=f"Base Intensity: {obj.get('base_intensity', 0):.4f}").pack(anchor='w', padx=4, pady=2)
        
        if obj.get('type') in ('gaussian_blobs', 'dots'):
            ttk.Label(props_frame, text=f"Sigma: {obj.get('spot_sigma', 0):.2f} px").pack(anchor='w', padx=4, pady=2)
        else:
            ttk.Label(props_frame, text=f"Size: {obj.get('size_px', 0):.2f} px").pack(anchor='w', padx=4, pady=2)
        
        # Composition
        comp_frame = ttk.LabelFrame(self.details_content, text="Fluorophore Composition")
        comp_frame.pack(fill=tk.X, padx=4, pady=4)
        
        composition = obj.get('composition', [])
        for comp in composition:
            fluor_idx = comp['fluor_index']
            fluor_name = fluor_names[fluor_idx] if fluor_idx < len(fluor_names) else f"F{fluor_idx+1}"
            ratio = comp['ratio']
            intensity = comp['intensity']
            
            comp_row = ttk.Frame(comp_frame)
            comp_row.pack(fill=tk.X, padx=4, pady=2)
            
            ttk.Label(comp_row, text=f"{fluor_name}:", width=10).pack(side=tk.LEFT)
            ttk.Label(comp_row, text=f"{ratio:.1%} ({intensity:.4f})").pack(side=tk.LEFT, padx=4)
            
            # Progress bar for ratio
            progress = ttk.Progressbar(comp_row, length=100, mode='determinate', maximum=100)
            progress['value'] = ratio * 100
            progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        
        # Region info
        region = obj.get('region', {})
        region_type = region.get('type', 'full')
        if region_type != 'full':
            region_frame = ttk.LabelFrame(self.details_content, text="Region")
            region_frame.pack(fill=tk.X, padx=4, pady=4)
            
            if region_type == 'rect':
                ttk.Label(region_frame, text=f"Rectangle: ({region.get('x0')}, {region.get('y0')}) "
                         f"{region.get('w')}×{region.get('h')}").pack(anchor='w', padx=4, pady=2)
            elif region_type == 'circle':
                ttk.Label(region_frame, text=f"Circle: center ({region.get('cx'):.1f}, {region.get('cy'):.1f}) "
                         f"r={region.get('r'):.1f}").pack(anchor='w', padx=4, pady=2)
        
        # Spectral profile
        self._add_spectral_profile(obj)
    
    def _show_multiple_objects_summary(self):
        """Show summary for multiple selected objects."""
        self.details_label.config(text=f"{len(self.selected_objects)} objects selected")
        
        summary_frame = ttk.LabelFrame(self.details_content, text="Selection Summary")
        summary_frame.pack(fill=tk.X, padx=4, pady=4)
        
        # Get selected objects from DataFrame
        selected_df = self.df[self.df['id'].isin(self.selected_objects)]
        
        if selected_df.empty:
            return
        
        # Calculate statistics using pandas
        ttk.Label(summary_frame, text=f"Count: {len(selected_df)}").pack(anchor='w', padx=4, pady=2)
        
        intensities = selected_df['base_intensity']
        if not intensities.empty:
            ttk.Label(summary_frame, text=f"Intensity: min={intensities.min():.3f}, "
                     f"max={intensities.max():.3f}, mean={intensities.mean():.3f}").pack(anchor='w', padx=4, pady=2)
        
        # Type distribution
        type_counts = selected_df['type'].value_counts()
        
        type_frame = ttk.LabelFrame(self.details_content, text="Type Distribution")
        type_frame.pack(fill=tk.X, padx=4, pady=4)
        for obj_type, count in type_counts.items():
            ttk.Label(type_frame, text=f"{obj_type}: {count}").pack(anchor='w', padx=4, pady=2)
    
    def _sort_by(self, column: str):
        """Sort objects by column."""
        # TODO: Implement sorting with pandas
        pass
    
    def _clear_selection(self):
        """Clear current selection."""
        # Unlock selection when explicitly clearing
        self.selection_locked = False
        self.lock_selection.set(False)
        
        self.tree.selection_remove(self.tree.selection())
        self.selected_objects = []
        self._update_stats()
        self._show_object_details()
    
    def set_selected_objects(self, object_ids: List[int]):
        """Programmatically select objects by their IDs (optimized).
        
        Args:
            object_ids: List of object IDs to select
        """
        # Set selection lock and flag when programmatically selecting
        self.selection_locked = True
        self.lock_selection.set(True)
        
        # Temporarily unbind selection event to prevent multiple triggers
        # This prevents _on_object_select from being called for each item added
        self.tree.unbind('<<TreeviewSelect>>')
        
        try:
            # Clear current selection
            self.tree.selection_remove(self.tree.selection())
            
            # Build ID→item mapping ONCE (O(n) instead of O(n²))
            # This is critical for performance with large selections
            id_to_item = {}
            for item in self.tree.get_children():
                obj_id = self.tree.item(item)['values'][0]
                id_to_item[obj_id] = item
            
            # Find items using hash lookup (O(1) per ID)
            items_to_select = []
            for obj_id in object_ids:
                if obj_id in id_to_item:
                    items_to_select.append(id_to_item[obj_id])
            
            # Select all items at once (more efficient than adding one by one)
            if items_to_select:
                self.tree.selection_set(items_to_select)
                # Scroll to make the first selected item visible
                self.tree.see(items_to_select[0])
            
            # Update internal state
            self.selected_objects = object_ids
            self._update_stats()
            self._show_object_details()
            
        finally:
            # Always rebind the event handler, even if an error occurs
            self.tree.bind('<<TreeviewSelect>>', self._on_object_select)
    
    def _export_csv(self):
        """Export filtered object list to CSV using pandas."""
        if self.filtered_df.empty:
            messagebox.showinfo("No Objects", "No objects to export.")
            return
        
        from tkinter import filedialog
        filepath = filedialog.asksaveasfilename(
            title='Export Objects',
            defaultextension='.csv',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
        )
        
        if not filepath:
            return
        
        try:
            fluor_names = self.get_fluorophore_names()
            
            # Prepare export data
            export_data = []
            for idx, row in self.filtered_df.iterrows():
                pos = row['position']
                obj_type = row['type']
                size = row.get('spot_sigma' if obj_type in ('gaussian_blobs', 'dots') else 'size_px', 0)
                
                comp = row['composition']
                fluor_list = []
                ratio_list = []
                for c in comp:
                    fname = fluor_names[c['fluor_index']] if c['fluor_index'] < len(fluor_names) else f"F{c['fluor_index']+1}"
                    fluor_list.append(fname)
                    ratio_list.append(f"{c['ratio']:.4f}")
                
                export_data.append({
                    'ID': row['id'],
                    'Type': obj_type,
                    'Position_Y': f"{pos[0]:.4f}",
                    'Position_X': f"{pos[1]:.4f}",
                    'Base_Intensity': f"{row['base_intensity']:.4f}",
                    'Size_or_Sigma': f"{size:.4f}",
                    'Fluorophores': '+'.join(fluor_list),
                    'Composition_Ratios': '+'.join(ratio_list)
                })
            
            # Use pandas to export
            export_df = pd.DataFrame(export_data)
            export_df.to_csv(filepath, index=False)
            
            messagebox.showinfo("Export Complete", f"Exported {len(export_data)} objects to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Export Failed", f"Failed to export objects:\n{str(e)}")
    
    def _add_zoomed_composite_view(self, obj: Dict[str, Any]):
        """Add a larger zoomed composite view for the selected object."""
        data = self.get_data()
        if not data or not data.has_data:
            return
        
        pos = obj.get('position', (0, 0))
        center_y, center_x = pos
        crop_size = 30  # Larger view for full inspector
        
        # Create zoom frame first so it's available in exception handler
        zoom_frame = ttk.LabelFrame(self.details_content, text="Zoomed View")
        zoom_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        try:
            # Generate zoomed composite using RGBRenderer
            from ..views.composite.rendering.rgb_renderer import RGBRenderer
            from ..views.composite.visual_settings import VisualSettingsManager
            
            Y = data.Y
            M = data.M
            H, W = data.field.shape
            
            # Create visual settings manager with default settings
            visual_settings = VisualSettingsManager(self.details_content)
            visual_settings.normalization_mode.set("global")
            visual_settings.gamma_correction.set(1.0)
            visual_settings.use_log_scaling.set(False)
            visual_settings.percentile_threshold.set(99.0)
            
            # Create RGB renderer and generate composite
            renderer = RGBRenderer(visual_settings)
            channels = [True] * Y.shape[0]  # Use all channels
            full_rgb = renderer.render_composite(data, channels)
            
            # Crop region
            y0 = int(max(0, center_y - crop_size))
            y1 = int(min(H, center_y + crop_size))
            x0 = int(max(0, center_x - crop_size))
            x1 = int(min(W, center_x + crop_size))
            
            cropped = full_rgb[y0:y1, x0:x1]
            
            # Create figure for zoomed view
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            fig = Figure(figsize=(4, 4), dpi=80)
            ax = fig.add_subplot(111)
            ax.imshow(cropped, origin='upper')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Draw crosshair at center
            h, w = cropped.shape[:2]
            rel_y = center_y - y0
            rel_x = center_x - x0
            ax.plot([rel_x], [rel_y], '+', color='yellow', markersize=12, markeredgewidth=2)
            ax.plot([rel_x], [rel_y], 'o', markerfacecolor='none', markeredgecolor='cyan',
                   markersize=10, markeredgewidth=2)
            
            # Draw bounding box based on object type/size
            obj_type = obj.get('type', 'unknown')
            size_px = obj.get('size_px', 3.0)
            spot_sigma = obj.get('spot_sigma', 2.0)
            
            # Determine box size based on object type
            if obj_type == 'circles' or obj_type == 'boxes':
                box_radius = size_px
            elif obj_type in ('gaussian_blobs', 'dots'):
                box_radius = 2.0 * spot_sigma  # 95% containment
            else:
                box_radius = 3.0  # Default
            
            # Draw rectangle around object
            from matplotlib.patches import Rectangle
            box_x = rel_x - box_radius
            box_y = rel_y - box_radius
            box_width = 2 * box_radius
            box_height = 2 * box_radius
            
            rect = Rectangle(
                (box_x, box_y), box_width, box_height,
                linewidth=1.5, edgecolor='lime', facecolor='none',
                linestyle='--', alpha=0.8
            )
            ax.add_patch(rect)
            
            ax.set_title(f"Local View ({2*crop_size}×{2*crop_size} px) | Type: {obj_type}", fontsize=10)
            fig.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, zoom_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
            
        except Exception as e:
            # Fallback if view generation fails
            ttk.Label(zoom_frame, text=f"Could not generate view: {str(e)}", 
                     foreground="gray").pack(pady=10)
    
    def _add_spectral_profile(self, obj: Dict[str, Any]):
        """Add spectral profile plot for the object's fluorophore composition."""
        data = self.get_data()
        if not data or not data.has_data or not data.spectral:
            return
        
        composition = obj.get('composition', [])
        if not composition:
            return
        
        try:
            spectral = data.spectral
            lambdas = spectral.lambdas
            fluor_names = self.get_fluorophore_names()
            
            # Calculate combined spectrum
            combined_spectrum = np.zeros_like(lambdas)
            individual_spectra = []
            
            for comp in composition:
                fluor_idx = comp['fluor_index']
                ratio = comp['ratio']
                
                if fluor_idx < len(spectral.fluors):
                    fluor = spectral.fluors[fluor_idx]
                    # Get normalized PDF for this fluorophore
                    pdf = spectral._pdf(fluor)
                    # Weight by ratio
                    weighted_pdf = pdf * ratio
                    combined_spectrum += weighted_pdf
                    
                    # Store for individual plotting
                    fluor_name = fluor_names[fluor_idx] if fluor_idx < len(fluor_names) else f"F{fluor_idx+1}"
                    individual_spectra.append((fluor_name, pdf, ratio))
            
            # Create spectral profile frame
            spectrum_frame = ttk.LabelFrame(self.details_content, text="Spectral Profile")
            spectrum_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
            
            # Create matplotlib figure
            fig = Figure(figsize=(5, 3), dpi=80)
            ax = fig.add_subplot(111)
            
            # Plot individual fluorophore spectra (normalized)
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
            for idx, (name, pdf, ratio) in enumerate(individual_spectra):
                pdf_norm = pdf / (np.max(pdf) + 1e-9)
                color = colors[idx % len(colors)]
                ax.plot(lambdas, pdf_norm, '--', color=color, alpha=0.6, linewidth=1.5,
                       label=f'{name} ({ratio:.1%})')
            
            # Plot combined spectrum (bold)
            if np.max(combined_spectrum) > 0:
                combined_norm = combined_spectrum / np.max(combined_spectrum)
                ax.plot(lambdas, combined_norm, 'k-', linewidth=2.5, 
                       label='Combined', zorder=10)
            
            ax.set_xlabel('Wavelength (nm)', fontsize=9)
            ax.set_ylabel('Normalized Intensity', fontsize=9)
            ax.set_title('Object Spectral Profile', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=8, loc='best')
            ax.tick_params(labelsize=8)
            
            fig.tight_layout()
            
            # Add canvas to frame
            canvas = FigureCanvasTkAgg(fig, spectrum_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
            
        except Exception as e:
            # Show error message if spectral profile generation fails
            error_label = ttk.Label(self.details_content, 
                                   text=f"Could not generate spectral profile: {str(e)}", 
                                   foreground="gray")
            error_label.pack(pady=10)

