"""Object layers manager for the spectral visualization GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import copy
import random
from .object_templates import TemplateManager, ObjectTemplate, FluorophoreComponent


class ObjectLayersManager:
    """Manager for object layers that can be placed on the image."""
    
    def __init__(self, parent_frame, log_callback, get_image_dims_callback, get_fluorophore_names_callback):
        self.parent_frame = parent_frame
        self.log = log_callback
        self.get_image_dims = get_image_dims_callback
        self.get_fluorophore_names = get_fluorophore_names_callback
        self.objects = []
        self.include_base_field = tk.BooleanVar(value=True)
        self.template_manager = TemplateManager()
        
        # Object editor variables
        self.obj_fluor = tk.StringVar(value="F1")  # Stores fluorophore name (e.g., "F1")
        self.obj_kind = tk.StringVar(value="gaussian_blobs")
        self.obj_count = tk.IntVar(value=50)
        self.obj_size = tk.DoubleVar(value=6.0)
        self.obj_i_min = tk.DoubleVar(value=0.5)
        self.obj_i_max = tk.DoubleVar(value=1.5)
        self.obj_sigma = tk.DoubleVar(value=2.0)
        self.obj_region_type = tk.StringVar(value="full")
        self.obj_x0 = tk.IntVar(value=0)
        self.obj_y0 = tk.IntVar(value=0)
        self.obj_w = tk.IntVar(value=64)
        self.obj_h = tk.IntVar(value=64)
        self.obj_cx = tk.DoubleVar(value=64)
        self.obj_cy = tk.DoubleVar(value=64)
        self.obj_r = tk.DoubleVar(value=40)
        
        # Composition mode: 'single' or 'template'
        self.composition_mode = tk.StringVar(value="single")
        self.composition_template = tk.StringVar(value="F1 Only")
        
        self._build_ui()
        self._add_preset_objects()
        
    def _build_ui(self):
        """Build the object layers management UI."""
        # Help text only
        help_frame = ttk.Frame(self.parent_frame)
        help_frame.pack(fill=tk.X, padx=4, pady=(4,0))
        ttk.Label(help_frame, text="Place specific fluorophores in regions", foreground="gray", font=('TkDefaultFont', 8)).pack(side=tk.LEFT)

        obj_main = ttk.Frame(self.parent_frame)
        obj_main.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        obj_main.columnconfigure(0, weight=1)

        # Object list with improved buttons - no bold label
        list_frame = ttk.Frame(obj_main)
        list_frame.grid(row=0, column=0, sticky="nsew", pady=(0,4))
        list_frame.columnconfigure(0, weight=1)
        
        header_frame = ttk.Frame(list_frame)
        header_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0,4))
        header_frame.columnconfigure(0, weight=1)
        
        # All buttons in a single row
        btns = ttk.Frame(header_frame)
        btns.pack(side=tk.LEFT)
        ttk.Button(btns, text="Add", width=6, command=self._add_object).pack(side=tk.LEFT, padx=(0,2))
        ttk.Button(btns, text="Remove", width=6, command=self._remove_object).pack(side=tk.LEFT, padx=(0,2))
        ttk.Button(btns, text="Copy", width=6, command=self._duplicate_object).pack(side=tk.LEFT, padx=(0,2))
        ttk.Button(btns, text="Quick-Assign", width=12, command=self._quick_assign_sample).pack(side=tk.LEFT)

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
        
        self._build_properties_tab(editor_notebook)
        self._build_region_tab(editor_notebook)
        
    def _build_properties_tab(self, notebook):
        """Build the properties tab for object editing with template support."""
        basic_tab = ttk.Frame(notebook)
        notebook.add(basic_tab, text="Properties")
        
        # Row 0: Composition Mode Selection
        mode_frame = ttk.LabelFrame(basic_tab, text="Composition Mode")
        mode_frame.grid(row=0, column=0, columnspan=4, sticky='ew', pady=(0,8))
        
        ttk.Radiobutton(mode_frame, text="Single Fluorophore", 
                       variable=self.composition_mode, value="single",
                       command=self._on_composition_mode_change).pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(mode_frame, text="Multi-Fluorophore Template", 
                       variable=self.composition_mode, value="template",
                       command=self._on_composition_mode_change).pack(side=tk.LEFT, padx=4)
        
        # Row 1: Fluorophore/Template Selection (dynamic based on mode)
        self.fluor_selection_frame = ttk.Frame(basic_tab)
        self.fluor_selection_frame.grid(row=1, column=0, columnspan=4, sticky='ew', pady=(0,8))
        
        # Single fluorophore selector
        self.single_fluor_frame = ttk.Frame(self.fluor_selection_frame)
        ttk.Label(self.single_fluor_frame, text="Fluorophore:").pack(side=tk.LEFT, padx=(0,4))
        self.fluor_combo = ttk.Combobox(self.single_fluor_frame, textvariable=self.obj_fluor, 
                                       values=self._get_fluorophore_list(), 
                                       state='readonly', width=15)
        self.fluor_combo.pack(side=tk.LEFT)
        self.fluor_combo.bind('<<ComboboxSelected>>', lambda e: self._auto_save())
        
        # Template selector
        self.template_frame = ttk.Frame(self.fluor_selection_frame)
        ttk.Label(self.template_frame, text="Template:").pack(side=tk.LEFT, padx=(0,4))
        self.template_combo = ttk.Combobox(self.template_frame, 
                                          textvariable=self.composition_template,
                                          values=self.template_manager.get_template_names(),
                                          state='readonly', width=20)
        self.template_combo.pack(side=tk.LEFT, padx=(0,4))
        self.template_combo.bind('<<ComboboxSelected>>', lambda e: self._auto_save())
        
        ttk.Button(self.template_frame, text="Manage Templates", 
                  command=self._open_template_manager, width=15).pack(side=tk.LEFT)
        
        # Row 2: Kind
        ttk.Label(basic_tab, text="Kind:").grid(row=2, column=0, sticky='w', padx=(0,4))
        self.kind_combo = ttk.Combobox(basic_tab, textvariable=self.obj_kind, 
                                       values=["circles","boxes","gaussian_blobs","dots"], 
                                       state='readonly', width=15)
        self.kind_combo.grid(row=2, column=1, columnspan=3, sticky='w')
        self.kind_combo.bind('<<ComboboxSelected>>', lambda e: (self._on_kind_change(), self._auto_save()))

        # Row 3: Count and Size
        ttk.Label(basic_tab, text="Count:").grid(row=3, column=0, sticky='w', pady=(6,0))
        count_entry = ttk.Entry(basic_tab, textvariable=self.obj_count, width=10)
        count_entry.grid(row=3, column=1, sticky='w', pady=(6,0))
        count_entry.bind('<KeyRelease>', lambda e: self._auto_save())
        
        self.size_label = ttk.Label(basic_tab, text="Size (px):")
        self.size_label.grid(row=3, column=2, sticky='w', padx=(12,4), pady=(6,0))
        self.size_entry = ttk.Entry(basic_tab, textvariable=self.obj_size, width=10)
        self.size_entry.grid(row=3, column=3, sticky='w', pady=(6,0))
        self.size_entry.bind('<KeyRelease>', lambda e: self._auto_save())

        # Row 4: Intensity range
        ttk.Label(basic_tab, text="Intensity:").grid(row=4, column=0, sticky='w', pady=(6,0))
        intensity_frame = ttk.Frame(basic_tab)
        intensity_frame.grid(row=4, column=1, columnspan=3, sticky='w', pady=(6,0))
        i_min_entry = ttk.Entry(intensity_frame, textvariable=self.obj_i_min, width=6)
        i_min_entry.pack(side=tk.LEFT)
        i_min_entry.bind('<KeyRelease>', lambda e: self._auto_save())
        ttk.Label(intensity_frame, text="to").pack(side=tk.LEFT, padx=2)
        i_max_entry = ttk.Entry(intensity_frame, textvariable=self.obj_i_max, width=6)
        i_max_entry.pack(side=tk.LEFT)
        i_max_entry.bind('<KeyRelease>', lambda e: self._auto_save())
        
        # Row 5: Spot sigma (for Gaussian blobs and dots only)
        self.sigma_label = ttk.Label(basic_tab, text="Spot σ (px):")
        self.sigma_label.grid(row=5, column=0, sticky='w', pady=(6,0))
        self.sigma_entry = ttk.Entry(basic_tab, textvariable=self.obj_sigma, width=10)
        self.sigma_entry.grid(row=5, column=1, columnspan=3, sticky='w', pady=(6,0))
        self.sigma_entry.bind('<KeyRelease>', lambda e: self._auto_save())
        
        # Initialize mode UI
        self._on_composition_mode_change()

    def _build_region_tab(self, notebook):
        """Build the region tab for object editing."""
        region_tab = ttk.Frame(notebook)
        notebook.add(region_tab, text="Region")
        
        ttk.Label(region_tab, text="Type:").grid(row=0, column=0, sticky='w')
        region_combo = ttk.Combobox(region_tab, textvariable=self.obj_region_type, values=["full","rect","circle"], state='readonly', width=12)
        region_combo.grid(row=0, column=1, sticky='w', padx=(0,8))
        region_combo.bind('<<ComboboxSelected>>', lambda e: (self._on_region_type_change(), self._auto_save()))

        # Container for dynamic region parameters
        self.region_params_frame = ttk.Frame(region_tab)
        self.region_params_frame.grid(row=1, column=0, columnspan=4, sticky='ew', pady=(6,0))
        
        # Initialize with full region (no parameters)
        self._update_region_ui()

    def _add_object(self, template_name=None):
        """Add a new object to the list."""
        # Use current image dimensions for sensible defaults
        H, W = self.get_image_dims()
        obj = {
            'kind': 'gaussian_blobs',
            'region': {'type': 'full'},  # Default to full/global region
            'count': 25,
            'size_px': max(3.0, min(W, H) / 20),  # Scale with image size
            'intensity_min': 0.5,
            'intensity_max': 1.5,
            'spot_sigma': max(1.5, min(W, H) / 40),
        }
        
        # Set composition based on template or default to single fluorophore
        if template_name:
            obj['template_name'] = template_name
            fluor_desc = template_name
        else:
            obj['fluor_index'] = len(self.objects) % max(1, 3)  # Cycle through 3 fluorophores by default
            fluor_desc = f"F{obj['fluor_index']+1}"
        
        self.objects.append(obj)
        self._refresh_object_list()
        
        # Auto-select the new object
        items = self.obj_tree.get_children()
        if items:
            self.obj_tree.selection_set(items[-1])
            self._on_object_select()
        
        self.log(f"Added object {len(self.objects)}: {fluor_desc}, {obj['kind']}")

    def _remove_object(self):
        """Remove selected object from the list."""
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
        """Duplicate selected object."""
        sel = self.obj_tree.selection()
        if not sel:
            return
        idx = self.obj_tree.index(sel[0])
        if 0 <= idx < len(self.objects):
            self.objects.append(copy.deepcopy(self.objects[idx]))
            self._refresh_object_list()

    def _refresh_object_list(self):
        """Refresh the object list display."""
        # Clear tree
        for i in self.obj_tree.get_children():
            self.obj_tree.delete(i)
        # Populate with simplified format
        for obj in self.objects:
            # Get fluorophore name/composition
            if 'template_name' in obj:
                fluor_name = f"[{obj['template_name']}]"
            else:
                fluor_idx = obj.get('fluor_index', 0)
                fluor_name = self._fluorophore_index_to_name(fluor_idx)
            
            kind = obj.get('kind', '')
            region = obj.get('region', {'type': 'full'})
            rtxt = region.get('type', 'full')
            if rtxt == 'rect':
                rtxt = f"rect({region.get('w',0)}×{region.get('h',0)})"
            elif rtxt == 'circle':
                rtxt = f"circle(r={region.get('r',0):.0f})"
            count = obj.get('count', 0)
            self.obj_tree.insert('', 'end', values=(fluor_name, kind, rtxt, count))
        
    def _on_object_select(self, event=None):
        """Handle object selection."""
        sel = self.obj_tree.selection()
        if not sel:
            return
        idx = self.obj_tree.index(sel[0])
        if not (0 <= idx < len(self.objects)):
            return
        obj = self.objects[idx]
        
        # Update fluorophore dropdown with current fluorophore list
        self.fluor_combo.config(values=self._get_fluorophore_list())
        
        # Check if object uses template or single fluorophore
        if 'template_name' in obj:
            # Template mode
            self.composition_mode.set("template")
            self.composition_template.set(obj['template_name'])
            self._on_composition_mode_change()
        else:
            # Single fluorophore mode
            self.composition_mode.set("single")
            fluor_idx = int(obj.get('fluor_index', 0))
            fluor_name = self._fluorophore_index_to_name(fluor_idx)
            self.obj_fluor.set(fluor_name)
            self._on_composition_mode_change()
        
        # Update other properties
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
        
        # Update kind-specific UI elements
        self._on_kind_change()

    def _auto_save(self):
        """Automatically save changes to the selected object."""
        sel = self.obj_tree.selection()
        if not sel:
            return
        idx = self.obj_tree.index(sel[0])
        if not (0 <= idx < len(self.objects)):
            return
        
        try:
            region_type = self.obj_region_type.get()
            region = {'type': region_type}
            if region_type == 'rect':
                region.update({'x0': self.obj_x0.get(), 'y0': self.obj_y0.get(), 
                              'w': self.obj_w.get(), 'h': self.obj_h.get()})
            elif region_type == 'circle':
                region.update({'cx': self.obj_cx.get(), 'cy': self.obj_cy.get(), 
                              'r': self.obj_r.get()})
            
            # Handle composition mode
            if self.composition_mode.get() == "template":
                template_name = self.composition_template.get()
                self.objects[idx] = {
                    'template_name': template_name,
                    'kind': self.obj_kind.get(),
                    'region': region,
                    'count': int(self.obj_count.get()),
                    'size_px': float(self.obj_size.get()),
                    'intensity_min': float(self.obj_i_min.get()),
                    'intensity_max': float(self.obj_i_max.get()),
                    'spot_sigma': float(self.obj_sigma.get()),
                }
            else:
                # Single fluorophore mode
                fluor_name = self.obj_fluor.get()
                fluor_index = self._fluorophore_name_to_index(fluor_name)
                
                self.objects[idx] = {
                    'fluor_index': fluor_index,
                    'kind': self.obj_kind.get(),
                    'region': region,
                    'count': int(self.obj_count.get()),
                    'size_px': float(self.obj_size.get()),
                    'intensity_min': float(self.obj_i_min.get()),
                    'intensity_max': float(self.obj_i_max.get()),
                    'spot_sigma': float(self.obj_sigma.get()),
                }
            self._refresh_object_list()
            
            # Re-select the updated item
            items = self.obj_tree.get_children()
            if idx < len(items):
                self.obj_tree.selection_set(items[idx])
            
        except Exception:
            pass  # Silently fail on invalid input during typing

    def _on_kind_change(self, event=None):
        """Handle object kind change to show/hide relevant parameters."""
        kind = self.obj_kind.get()
        
        # Show/hide parameters based on object kind
        if kind in ("gaussian_blobs", "dots"):
            # Gaussian blobs and dots: ONLY use spot_sigma (size_px is ignored in code!)
            self.size_label.grid_remove()
            self.size_entry.grid_remove()
            
            # Show sigma controls
            self.sigma_label.grid(row=3, column=0, sticky='w', pady=(6,0))
            self.sigma_entry.grid(row=3, column=1, columnspan=3, sticky='w', pady=(6,0))
        else:
            # Circles and boxes: use size_px for radius/dimensions, no sigma
            self.size_label.grid(row=1, column=2, sticky='w', padx=(12,4), pady=(6,0))
            self.size_entry.grid(row=1, column=3, sticky='w', pady=(6,0))
            self.sigma_label.grid_remove()
            self.sigma_entry.grid_remove()
    
    def _on_composition_mode_change(self):
        """Handle composition mode change between single/template."""
        mode = self.composition_mode.get()
        
        # Hide both frames
        self.single_fluor_frame.pack_forget()
        self.template_frame.pack_forget()
        
        # Show appropriate frame
        if mode == "single":
            self.single_fluor_frame.pack(fill=tk.X)
        else:  # template
            self.template_frame.pack(fill=tk.X)
            # Refresh template list
            self.template_combo.config(values=self.template_manager.get_template_names())
    
    def _open_template_manager(self):
        """Open template manager dialog."""
        dialog = tk.Toplevel(self.parent_frame)
        dialog.title("Manage Multi-Fluorophore Templates")
        dialog.geometry("700x550")
        dialog.transient()
        
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Template list
        list_frame = ttk.LabelFrame(main_frame, text="Available Templates")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0,10))
        
        # Listbox with scrollbar
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        template_listbox = tk.Listbox(list_container, yscrollcommand=scrollbar.set, height=10)
        template_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=template_listbox.yview)
        
        # Populate template list
        def refresh_list():
            template_listbox.delete(0, tk.END)
            for template in self.template_manager.templates:
                # Show composition summary
                comp_summary = ", ".join([f"F{c.fluor_index+1}:{c.ratio:.0%}" 
                                         for c in template.composition])
                template_listbox.insert(tk.END, f"{template.name} ({comp_summary})")
        
        refresh_list()
        
        # Template details frame
        detail_frame = ttk.LabelFrame(main_frame, text="Template Details")
        detail_frame.pack(fill=tk.X, pady=(0,10))
        
        detail_text = tk.Text(detail_frame, height=6, width=60, state='disabled')
        detail_text.pack(padx=5, pady=5)
        
        def show_template_details(event=None):
            """Show details of selected template."""
            selection = template_listbox.curselection()
            if not selection:
                return
            
            idx = selection[0]
            if idx >= len(self.template_manager.templates):
                return
            
            template = self.template_manager.templates[idx]
            
            details = f"Name: {template.name}\n"
            details += f"Description: {template.description}\n\n"
            details += "Composition:\n"
            for comp in template.composition:
                fluor_name = self._fluorophore_index_to_name(comp.fluor_index)
                details += f"  • {fluor_name}: {comp.ratio:.1%} "
                if comp.ratio_noise > 0:
                    details += f"(±{comp.ratio_noise:.1%} noise)"
                details += "\n"
            
            detail_text.config(state='normal')
            detail_text.delete('1.0', tk.END)
            detail_text.insert('1.0', details)
            detail_text.config(state='disabled')
        
        template_listbox.bind('<<ListboxSelect>>', show_template_details)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Create New", 
                  command=lambda: self._create_custom_template(dialog, refresh_list)).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Delete Selected", 
                  command=lambda: self._delete_template(template_listbox, refresh_list)).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Close", command=dialog.destroy).pack(side=tk.RIGHT)
    
    def _create_custom_template(self, parent, refresh_callback):
        """Open dialog to create a custom template."""
        dialog = tk.Toplevel(parent)
        dialog.title("Create Custom Template")
        dialog.geometry("500x450")
        dialog.transient(parent)
        
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Template name
        ttk.Label(main_frame, text="Template Name:").grid(row=0, column=0, sticky='w', pady=5)
        name_var = tk.StringVar(value="Custom Mix")
        ttk.Entry(main_frame, textvariable=name_var, width=30).grid(row=0, column=1, sticky='w', pady=5)
        
        # Description
        ttk.Label(main_frame, text="Description:").grid(row=1, column=0, sticky='w', pady=5)
        desc_var = tk.StringVar(value="")
        ttk.Entry(main_frame, textvariable=desc_var, width=30).grid(row=1, column=1, sticky='w', pady=5)
        
        # Composition builder
        comp_frame = ttk.LabelFrame(main_frame, text="Fluorophore Composition")
        comp_frame.grid(row=2, column=0, columnspan=2, sticky='ew', pady=10)
        
        # List of components
        components = []  # List of (fluor_index, ratio, ratio_noise)
        
        comp_listbox = tk.Listbox(comp_frame, height=6)
        comp_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        def refresh_comp_list():
            comp_listbox.delete(0, tk.END)
            total_ratio = sum(c[1] for c in components)
            for fluor_idx, ratio, noise in components:
                fluor_name = self._fluorophore_index_to_name(fluor_idx)
                comp_listbox.insert(tk.END, 
                                  f"{fluor_name}: {ratio:.1%} ±{noise:.1%} (Total: {total_ratio:.1%})")
        
        # Add component controls
        add_frame = ttk.Frame(comp_frame)
        add_frame.pack(fill=tk.X, padx=5, pady=5)
        
        fluor_names = self._get_fluorophore_list()
        fluor_var = tk.StringVar(value=fluor_names[0] if fluor_names else "F1")
        ratio_var = tk.DoubleVar(value=0.5)
        noise_var = tk.DoubleVar(value=0.05)
        
        ttk.Label(add_frame, text="Fluorophore:").grid(row=0, column=0, sticky='w', padx=2)
        ttk.Combobox(add_frame, textvariable=fluor_var, values=fluor_names, 
                    state='readonly', width=10).grid(row=0, column=1, padx=2)
        
        ttk.Label(add_frame, text="Ratio:").grid(row=0, column=2, sticky='w', padx=2)
        ttk.Entry(add_frame, textvariable=ratio_var, width=8).grid(row=0, column=3, padx=2)
        
        ttk.Label(add_frame, text="Noise:").grid(row=1, column=0, sticky='w', padx=2)
        ttk.Entry(add_frame, textvariable=noise_var, width=8).grid(row=1, column=1, padx=2)
        
        def add_component():
            fluor_idx = self._fluorophore_name_to_index(fluor_var.get())
            components.append((fluor_idx, ratio_var.get(), noise_var.get()))
            refresh_comp_list()
        
        def remove_component():
            selection = comp_listbox.curselection()
            if selection and components:
                components.pop(selection[0])
                refresh_comp_list()
        
        ttk.Button(add_frame, text="Add", command=add_component).grid(row=1, column=2, padx=2)
        ttk.Button(add_frame, text="Remove", command=remove_component).grid(row=1, column=3, padx=2)
        
        # Save button
        def save_template():
            if not components:
                from tkinter import messagebox
                messagebox.showwarning("No Components", "Add at least one fluorophore component!")
                return
            
            template = ObjectTemplate(
                name=name_var.get(),
                description=desc_var.get(),
                composition=[FluorophoreComponent(f_idx, ratio, noise) 
                            for f_idx, ratio, noise in components]
            )
            
            self.template_manager.add_template(template)
            refresh_callback()
            self.log(f"Created template: {template.name}")
            dialog.destroy()
        
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        ttk.Button(button_frame, text="Save Template", command=save_template).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT)
    
    def _delete_template(self, listbox, refresh_callback):
        """Delete selected template."""
        selection = listbox.curselection()
        if not selection:
            return
        
        idx = selection[0]
        if idx >= len(self.template_manager.templates):
            return
        
        template = self.template_manager.templates[idx]
        
        # Don't allow deleting built-in single fluorophore templates
        if template.name.endswith(" Only"):
            from tkinter import messagebox
            messagebox.showinfo("Cannot Delete", "Built-in templates cannot be deleted.")
            return
        
        from tkinter import messagebox
        if messagebox.askyesno("Confirm Delete", f"Delete template '{template.name}'?"):
            self.template_manager.remove_template(template.name)
            refresh_callback()
            self.log(f"Deleted template: {template.name}")
            
    def _on_region_type_change(self, event=None):
        """Handle region type change."""
        # Update UI to show only relevant parameters
        self._update_region_ui()
        
        # Visual feedback when changing region type
        region_type = self.obj_region_type.get()
        if region_type == "full":
            self.log("Region set to full image")
        elif region_type == "rect":
            self.log("Region set to rectangle - configure x0, y0, width, height")
        elif region_type == "circle":
            self.log("Region set to circle - configure center and radius")
    
    def _update_region_ui(self):
        """Update region UI to show only relevant parameters."""
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
                entry = ttk.Entry(rect_frame, textvariable=var, width=6)
                entry.grid(row=0, column=i*2+1, sticky='w', padx=(0,8))
                entry.bind('<KeyRelease>', lambda e: self._auto_save())
                
        elif region_type == "circle":
            # Circle parameters only
            circle_frame = ttk.LabelFrame(self.region_params_frame, text="Circle Parameters")
            circle_frame.pack(fill='x', pady=(6,0))
            
            for i, (label, var) in enumerate([("center_x:", self.obj_cx), ("center_y:", self.obj_cy), ("radius:", self.obj_r)]):
                ttk.Label(circle_frame, text=label).grid(row=0, column=i*2, sticky='w', padx=(0,2))
                entry = ttk.Entry(circle_frame, textvariable=var, width=8)
                entry.grid(row=0, column=i*2+1, sticky='w', padx=(0,8))
                entry.bind('<KeyRelease>', lambda e: self._auto_save())
                
        else:  # "full"
            # No parameters needed for full image
            ttk.Label(self.region_params_frame, text="Full image - no parameters needed", 
                     foreground="gray", font=('TkDefaultFont', 9, 'italic')).pack(pady=10)
                     
    def get_objects(self):
        """Get list of object specifications with composition support."""
        # Convert objects to include composition if using templates
        converted_objects = []
        for obj in self.objects:
            obj_copy = obj.copy()
            
            # If object has a template_name, convert to composition format
            if 'template_name' in obj_copy:
                template = self.template_manager.get_template(obj_copy['template_name'])
                if template:
                    obj_copy['composition'] = template.get_composition_for_object()
                    # Remove fluor_index since we're using composition
                    obj_copy.pop('fluor_index', None)
                obj_copy.pop('template_name', None)  # Remove template_name before passing to spatial
            
            converted_objects.append(obj_copy)
        
        return converted_objects
        
    def should_include_base_field(self):
        """Check if base field should be included."""
        return self.include_base_field.get()
    
    def _get_fluorophore_list(self):
        """Get list of available fluorophore names."""
        try:
            fluor_names = self.get_fluorophore_names()
            return fluor_names if fluor_names else ["F1", "F2", "F3"]
        except:
            return ["F1", "F2", "F3"]
    
    def _fluorophore_name_to_index(self, name: str) -> int:
        """Convert fluorophore name to 0-indexed integer by looking up in fluorophore list."""
        try:
            fluor_names = self._get_fluorophore_list()
            if name in fluor_names:
                return fluor_names.index(name)
            # Fallback: try to extract number from "FX" format
            if name.startswith('F') and name[1:].isdigit():
                return int(name[1:]) - 1
            return 0
        except (ValueError, IndexError):
            return 0
    
    def _fluorophore_index_to_name(self, index: int) -> str:
        """Convert 0-indexed integer to fluorophore name by looking up in fluorophore list."""
        try:
            fluor_names = self._get_fluorophore_list()
            if 0 <= index < len(fluor_names):
                return fluor_names[index]
            return f"F{index+1}"  # Fallback
        except:
            return f"F{index+1}"
        
    def _add_preset_objects(self):
        """Add 3 preset objects, one for each default fluorophore."""
        H, W = self.get_image_dims()
        
        # Preset objects with different characteristics for each fluorophore
        presets = [
            {
                'fluor_index': 0,
                'kind': 'gaussian_blobs',
                'region': {'type': 'full'},
                'count': 30,
                'size_px': 4.0,
                'intensity_min': 0.8,
                'intensity_max': 1.2,
                'spot_sigma': 3.0,  # Larger sigma for visible Gaussian blobs
            },
            {
                'fluor_index': 1,
                'kind': 'circles',
                'region': {'type': 'full'},
                'count': 20,
                'size_px': 6.0,
                'intensity_min': 0.6,
                'intensity_max': 1.0,
                'spot_sigma': 1.5,  # Not used for circles, but set for consistency
            },
            {
                'fluor_index': 2,
                'kind': 'dots',
                'region': {'type': 'full'},
                'count': 40,
                'size_px': 3.0,
                'intensity_min': 0.4,
                'intensity_max': 0.8,
                'spot_sigma': 1.5,  # Moderate sigma for dot-like appearance
            }
        ]
        
        # Add preset objects
        for preset in presets:
            self.objects.append(preset)
            
        # Refresh the list to show the presets
        self._refresh_object_list()
        
        # Auto-select the first object
        items = self.obj_tree.get_children()
        if items:
            self.obj_tree.selection_set(items[0])
            self._on_object_select()
    
    def _quick_assign_sample(self):
        """Generate a sample dataset with all available fluorophores.
        
        Creates a varied distribution of Gaussian blobs with:
        - All available fluorophores
        - Varying spot counts (100-2000) across fluorophores
        - Random positions
        - Random amplitudes (0.5-1.5)
        - Random spot sigma (0.5-6.0)
        - One small contamination dot with low intensity
        """
        # Get image dimensions
        try:
            img_h, img_w = self.get_image_dims()
        except:
            self.log("Cannot determine image size. Using default 128x128.")
            img_h, img_w = 128, 128
        
        # Get available fluorophores
        fluorophore_names = self.get_fluorophore_names()
        if not fluorophore_names:
            from tkinter import messagebox
            messagebox.showwarning("No Fluorophores", "Please add fluorophores first!")
            return
        
        num_fluors = len(fluorophore_names)
        
        # Clear existing objects
        self.objects.clear()
        
        # Generate spot counts for each fluorophore
        # Using your existing random mechanism
        total_spots = 0
        for fluor_idx, fluor_name in enumerate(fluorophore_names):
            # Random count between 100-2000 for each fluorophore
            count = random.randint(100, 2000)
            
            # Add one object that generates 'count' spots
            amplitude_min = 0.5
            amplitude_max = 1.5
            spot_sigma = random.uniform(0.5, 6.0)
            
            obj = {
                'fluor_index': fluor_idx,
                'kind': 'gaussian_blobs',
                'region': {'type': 'full'},
                'count': count,
                'size_px': 0,  # Not used for Gaussian blobs
                'intensity_min': amplitude_min,
                'intensity_max': amplitude_max,
                'spot_sigma': spot_sigma
            }
            self.objects.append(obj)
            
            total_spots += count
        
        # Add contamination dot (small, low intensity, random fluorophore)
        contam_fluor_idx = random.randint(0, num_fluors - 1)
        contam_fluor_name = fluorophore_names[contam_fluor_idx]
        contam_sigma = random.uniform(0.3, 0.8)
        
        contam_obj = {
            'fluor_index': contam_fluor_idx,
            'kind': 'dots',
            'region': {'type': 'full'},
            'count': 1,
            'size_px': 0,
            'intensity_min': 0.1,
            'intensity_max': 0.3,
            'spot_sigma': contam_sigma
        }
        self.objects.append(contam_obj)
        
        # Refresh the list
        self._refresh_object_list()
        
        # Log summary
        self.log(f"Generated {total_spots + 1} spots across {num_fluors} fluorophores")
        self.log(f"   + 1 contamination dot ({contam_fluor_name}, σ={contam_sigma:.2f})")
