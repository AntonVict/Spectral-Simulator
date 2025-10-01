"""Object Inspector View - detailed information about individual generated objects."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Dict, Any, Optional, Callable
import numpy as np
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
        
        self.objects: List[Dict[str, Any]] = []
        self.filtered_objects: List[Dict[str, Any]] = []
        self.selected_objects: List[int] = []  # List of object IDs
        
        # Filter state
        self.filter_fluor = tk.StringVar(value="All")
        self.filter_type = tk.StringVar(value="All")
        self.filter_intensity_min = tk.DoubleVar(value=0.0)
        self.filter_intensity_max = tk.DoubleVar(value=10.0)
        self.search_var = tk.StringVar()
        self.search_var.trace('w', lambda *args: self._apply_filters())
        
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
        
        # Treeview with multiple columns
        columns = ('id', 'fluor', 'type', 'position', 'intensity')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', selectmode='extended')
        
        # Configure columns
        self.tree.heading('id', text='ID', command=lambda: self._sort_by('id'))
        self.tree.heading('fluor', text='Fluorophore', command=lambda: self._sort_by('fluor'))
        self.tree.heading('type', text='Type', command=lambda: self._sort_by('type'))
        self.tree.heading('position', text='Position (y,x)', command=lambda: self._sort_by('position'))
        self.tree.heading('intensity', text='Intensity', command=lambda: self._sort_by('intensity'))
        
        self.tree.column('id', width=50, stretch=False)
        self.tree.column('fluor', width=100, stretch=True)
        self.tree.column('type', width=100, stretch=False)
        self.tree.column('position', width=100, stretch=False)
        self.tree.column('intensity', width=80, stretch=False)
        
        self.tree.grid(row=0, column=0, sticky='nsew')
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        # Bind selection event
        self.tree.bind('<<TreeviewSelect>>', self._on_object_select)
        
    def _build_filters(self, parent: ttk.Frame):
        """Build filter controls."""
        # Row 0: Search
        ttk.Label(parent, text="Search:").grid(row=0, column=0, sticky='w', padx=2, pady=2)
        ttk.Entry(parent, textvariable=self.search_var, width=15).grid(row=0, column=1, sticky='ew', padx=2, pady=2)
        
        # Row 0: Fluorophore filter
        ttk.Label(parent, text="Fluorophore:").grid(row=0, column=2, sticky='w', padx=(10,2), pady=2)
        self.fluor_combo = ttk.Combobox(parent, textvariable=self.filter_fluor, state='readonly', width=12)
        self.fluor_combo.grid(row=0, column=3, sticky='ew', padx=2, pady=2)
        self.fluor_combo.bind('<<ComboboxSelected>>', lambda e: self._apply_filters())
        
        # Row 1: Type filter
        ttk.Label(parent, text="Type:").grid(row=1, column=0, sticky='w', padx=2, pady=2)
        type_combo = ttk.Combobox(parent, textvariable=self.filter_type, 
                                  values=["All", "dots", "gaussian_blobs", "circles", "boxes"],
                                  state='readonly', width=12)
        type_combo.grid(row=1, column=1, sticky='ew', padx=2, pady=2)
        type_combo.bind('<<ComboboxSelected>>', lambda e: self._apply_filters())
        
        # Row 1: Intensity range
        ttk.Label(parent, text="Intensity:").grid(row=1, column=2, sticky='w', padx=(10,2), pady=2)
        intensity_frame = ttk.Frame(parent)
        intensity_frame.grid(row=1, column=3, sticky='ew', padx=2, pady=2)
        ttk.Entry(intensity_frame, textvariable=self.filter_intensity_min, width=6).pack(side=tk.LEFT)
        ttk.Label(intensity_frame, text="to").pack(side=tk.LEFT, padx=2)
        ttk.Entry(intensity_frame, textvariable=self.filter_intensity_max, width=6).pack(side=tk.LEFT)
        
        # Apply button
        ttk.Button(parent, text="Apply", command=self._apply_filters, width=8).grid(row=1, column=4, padx=4, pady=2)
        
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(3, weight=1)
    
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
            self.objects = []
            self.filtered_objects = []
            self._update_tree()
            self._update_stats()
            return
        
        # Get objects from metadata
        self.objects = data.metadata.get('objects', [])
        
        # Update fluorophore filter options
        fluor_names = self.get_fluorophore_names()
        self.fluor_combo.config(values=["All"] + fluor_names)
        
        # Apply current filters
        self._apply_filters()
        
    def _apply_filters(self):
        """Apply current filters to object list."""
        if not self.objects:
            self.filtered_objects = []
            self._update_tree()
            self._update_stats()
            return
        
        filtered = []
        search_text = self.search_var.get().lower()
        fluor_filter = self.filter_fluor.get()
        type_filter = self.filter_type.get()
        intensity_min = self.filter_intensity_min.get()
        intensity_max = self.filter_intensity_max.get()
        
        fluor_names = self.get_fluorophore_names()
        
        for obj in self.objects:
            # Search filter (ID or position)
            if search_text:
                obj_text = f"{obj['id']} {obj['position']}"
                if search_text not in obj_text.lower():
                    continue
            
            # Type filter
            if type_filter != "All" and obj.get('type', '') != type_filter:
                continue
            
            # Intensity filter
            if obj.get('base_intensity', 0) < intensity_min or obj.get('base_intensity', 0) > intensity_max:
                continue
            
            # Fluorophore filter
            if fluor_filter != "All":
                # Check if object has this fluorophore
                has_fluor = False
                try:
                    fluor_idx = fluor_names.index(fluor_filter)
                    for comp in obj.get('composition', []):
                        if comp['fluor_index'] == fluor_idx:
                            has_fluor = True
                            break
                except (ValueError, KeyError):
                    pass
                
                if not has_fluor:
                    continue
            
            filtered.append(obj)
        
        self.filtered_objects = filtered
        self._update_tree()
        self._update_stats()
    
    def _update_tree(self):
        """Update tree view with filtered objects."""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Get fluorophore names
        fluor_names = self.get_fluorophore_names()
        
        # Add filtered objects
        for obj in self.filtered_objects:
            obj_id = obj['id']
            
            # Get fluorophore composition string
            comp = obj.get('composition', [])
            if len(comp) == 1:
                fluor_str = fluor_names[comp[0]['fluor_index']] if comp[0]['fluor_index'] < len(fluor_names) else f"F{comp[0]['fluor_index']+1}"
            else:
                fluor_parts = []
                for c in comp:
                    fname = fluor_names[c['fluor_index']] if c['fluor_index'] < len(fluor_names) else f"F{c['fluor_index']+1}"
                    fluor_parts.append(f"{fname}({c['ratio']:.0%})")
                fluor_str = "+".join(fluor_parts)
            
            obj_type = obj.get('type', 'unknown')
            pos = obj.get('position', (0, 0))
            pos_str = f"({pos[0]:.1f}, {pos[1]:.1f})"
            intensity = obj.get('base_intensity', 0.0)
            
            self.tree.insert('', 'end', iid=str(obj_id), values=(obj_id, fluor_str, obj_type, pos_str, f"{intensity:.3f}"))
    
    def _update_stats(self):
        """Update statistics label."""
        total = len(self.objects)
        filtered = len(self.filtered_objects)
        selected = len(self.selected_objects)
        
        if total == 0:
            self.stats_label.config(text="No objects")
        elif filtered == total:
            self.stats_label.config(text=f"{total} objects | {selected} selected")
        else:
            self.stats_label.config(text=f"{filtered} of {total} objects | {selected} selected")
    
    def _on_object_select(self, event=None):
        """Handle object selection in tree."""
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
            obj = next((o for o in self.objects if o['id'] == obj_id), None)
            if obj:
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
        ttk.Label(props_frame, text=f"Position: ({pos[0]:.2f}, {pos[1]:.2f}) px").pack(anchor='w', padx=4, pady=2)
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
    
    def _show_multiple_objects_summary(self):
        """Show summary for multiple selected objects."""
        self.details_label.config(text=f"{len(self.selected_objects)} objects selected")
        
        summary_frame = ttk.LabelFrame(self.details_content, text="Selection Summary")
        summary_frame.pack(fill=tk.X, padx=4, pady=4)
        
        # Get selected objects
        selected_objs = [o for o in self.objects if o['id'] in self.selected_objects]
        
        # Calculate statistics
        intensities = [o.get('base_intensity', 0) for o in selected_objs]
        positions = [o.get('position', (0, 0)) for o in selected_objs]
        
        ttk.Label(summary_frame, text=f"Count: {len(selected_objs)}").pack(anchor='w', padx=4, pady=2)
        
        if intensities:
            ttk.Label(summary_frame, text=f"Intensity: min={min(intensities):.3f}, "
                     f"max={max(intensities):.3f}, mean={np.mean(intensities):.3f}").pack(anchor='w', padx=4, pady=2)
        
        # Type distribution
        types = {}
        for obj in selected_objs:
            t = obj.get('type', 'unknown')
            types[t] = types.get(t, 0) + 1
        
        type_frame = ttk.LabelFrame(self.details_content, text="Type Distribution")
        type_frame.pack(fill=tk.X, padx=4, pady=4)
        for t, count in types.items():
            ttk.Label(type_frame, text=f"{t}: {count}").pack(anchor='w', padx=4, pady=2)
    
    def _sort_by(self, column: str):
        """Sort objects by column."""
        # TODO: Implement sorting
        pass
    
    def _clear_selection(self):
        """Clear current selection."""
        self.tree.selection_remove(self.tree.selection())
        self.selected_objects = []
        self._update_stats()
        self._show_object_details()
    
    def _export_csv(self):
        """Export object list to CSV."""
        if not self.filtered_objects:
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
            import csv
            fluor_names = self.get_fluorophore_names()
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow(['ID', 'Type', 'Position_Y', 'Position_X', 'Base_Intensity', 
                               'Size_or_Sigma', 'Fluorophores', 'Composition_Ratios'])
                
                # Data
                for obj in self.filtered_objects:
                    pos = obj.get('position', (0, 0))
                    size = obj.get('spot_sigma' if obj.get('type') in ('gaussian_blobs', 'dots') else 'size_px', 0)
                    
                    comp = obj.get('composition', [])
                    fluor_list = []
                    ratio_list = []
                    for c in comp:
                        fname = fluor_names[c['fluor_index']] if c['fluor_index'] < len(fluor_names) else f"F{c['fluor_index']+1}"
                        fluor_list.append(fname)
                        ratio_list.append(f"{c['ratio']:.4f}")
                    
                    writer.writerow([
                        obj['id'],
                        obj.get('type', 'unknown'),
                        f"{pos[0]:.4f}",
                        f"{pos[1]:.4f}",
                        f"{obj.get('base_intensity', 0):.4f}",
                        f"{size:.4f}",
                        '+'.join(fluor_list),
                        '+'.join(ratio_list)
                    ])
            
            messagebox.showinfo("Export Complete", f"Exported {len(self.filtered_objects)} objects to:\n{filepath}")
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
        
        try:
            # Generate zoomed composite
            from ..views.composite.composite_view import create_composite_rgb
            
            Y = data.Y
            M = data.M
            H, W = data.field.shape
            
            # Create full composite
            full_rgb = create_composite_rgb(Y, M, H, W, 
                                           channel_selection=None,
                                           colormap='turbo',
                                           normalize_mode='global',
                                           gamma=1.0,
                                           brightness=1.0,
                                           contrast=1.0)
            
            # Crop region
            y0 = int(max(0, center_y - crop_size))
            y1 = int(min(H, center_y + crop_size))
            x0 = int(max(0, center_x - crop_size))
            x1 = int(min(W, center_x + crop_size))
            
            cropped = full_rgb[y0:y1, x0:x1]
            
            # Create figure for zoomed view
            zoom_frame = ttk.LabelFrame(self.details_content, text="Zoomed View")
            zoom_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
            
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
            
            ax.set_title(f"Local View ({2*crop_size}×{2*crop_size} px)", fontsize=10)
            fig.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, zoom_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
            
        except Exception as e:
            # Fallback if view generation fails
            ttk.Label(zoom_frame, text=f"Could not generate view: {str(e)}", 
                     foreground="gray").pack(pady=10)

