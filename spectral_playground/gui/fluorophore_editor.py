"""Fluorophore editor component for the spectral unmixing playground GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from spectral_playground.core.spectra import Fluorophore


class FluorophoreEditor(ttk.Frame):
    """Editor widget for configuring fluorophore spectral properties."""
    
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
        
        self._build_ui()
        
    def _build_ui(self):
        """Build the fluorophore editor UI."""
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
        """Handle model selection change."""
        self.data['model'] = self.model_var.get()
        self._build_params()
        # Don't auto-update anymore
        
    def _on_param_change(self, event=None):
        """Handle parameter value change."""
        # Don't auto-update anymore - require explicit apply
            
    def _build_params(self):
        """Build parameter input fields based on selected model."""
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
        """Add a parameter input field."""
        row = len(self.param_vars)
        ttk.Label(self.params_frame, text=label).grid(row=row, column=0, sticky="w", padx=(0,4))
        var = tk.DoubleVar(value=default_val)
        self.param_vars[key] = var
        entry = ttk.Entry(self.params_frame, textvariable=var, width=8)
        entry.grid(row=row, column=1, padx=2, pady=1)
        entry.bind('<KeyRelease>', self._on_param_change)
        
    def apply_changes(self):
        """Apply changes to the fluorophore data and notify parent."""
        self.data['model'] = self.model_var.get()
        self.data['brightness'] = self.brightness_var.get()
        self.data['params'] = {k: v.get() for k, v in self.param_vars.items()}
        if self.on_update:
            self.on_update(self.fluor_idx, self.data)
    
    def _update_data(self):
        """Legacy method - now just updates internal data without notifying parent."""
        self.data['model'] = self.model_var.get()
        self.data['brightness'] = self.brightness_var.get()
        self.data['params'] = {k: v.get() for k, v in self.param_vars.items()}
        
    def get_fluorophore(self):
        """Get a Fluorophore object from current data."""
        return Fluorophore(
            name=self.data['name'],
            model=self.data['model'],
            params=self.data['params'],
            brightness=self.data['brightness']
        )


class FluorophoreListManager:
    """Manager for the fluorophore list and editor."""
    
    def __init__(self, parent_frame, log_callback):
        self.parent_frame = parent_frame
        self.log = log_callback
        self.fluor_data = []
        self.current_editor = None
        self._build_ui()
        
    def _build_ui(self):
        """Build the fluorophore list management UI."""
        # List header with controls
        list_header = ttk.Frame(self.parent_frame)
        list_header.pack(fill=tk.X, pady=(0,4))
        ttk.Label(list_header, text="Fluorophore List:", font=('TkDefaultFont', 9, 'bold')).pack(side=tk.LEFT)
        
        btn_frame = ttk.Frame(list_header)
        btn_frame.pack(side=tk.RIGHT)
        ttk.Button(btn_frame, text="+ Add", command=self.add_fluorophore, width=8).pack(side=tk.LEFT, padx=(0,2))
        ttk.Button(btn_frame, text="- Remove", command=self.remove_fluorophore, width=8).pack(side=tk.LEFT)
        
        # Treeview for fluorophore list (more compact)
        tree_frame = ttk.Frame(self.parent_frame)
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
        editor_label = ttk.Label(self.parent_frame, text="Edit Selected Fluorophore:", font=('TkDefaultFont', 9, 'bold'))
        editor_label.pack(anchor='w', pady=(8,2))
        
        self.fluor_editor_frame = ttk.Frame(self.parent_frame, relief='groove', borderwidth=2)
        self.fluor_editor_frame.pack(fill=tk.X, pady=2)
        
        # Apply button for fluorophore changes
        apply_frame = ttk.Frame(self.parent_frame)
        apply_frame.pack(fill=tk.X, pady=(4,0))
        
        self.apply_btn = ttk.Button(apply_frame, text="Apply Changes", command=self._apply_fluor_changes)
        self.apply_btn.pack(side=tk.LEFT)
        ttk.Label(apply_frame, text="Select a fluorophore above to edit", foreground="gray", font=('TkDefaultFont', 8)).pack(side=tk.RIGHT)
        
        # Initialize with 3 fluorophores
        self.add_fluorophore()
        self.add_fluorophore()
        self.add_fluorophore()
        
    def add_fluorophore(self):
        """Add a new fluorophore to the list."""
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
        
    def remove_fluorophore(self):
        """Remove selected fluorophore from the list."""
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
        """Update the fluorophore list display."""
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
        """Format parameter display string."""
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
        """Handle fluorophore selection."""
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
            
    def _apply_fluor_changes(self):
        """Apply changes to the currently selected fluorophore."""
        if self.current_editor:
            try:
                self.current_editor.apply_changes()
                self.log("Applied fluorophore changes successfully")
            except Exception as e:
                self.log(f"Error applying fluorophore changes: {str(e)}")
        else:
            self.log("No fluorophore selected for editing")
            
    def _on_fluor_update(self, idx, data):
        """Handle fluorophore data update."""
        # Update the stored data
        if idx < len(self.fluor_data):
            self.fluor_data[idx] = data
            self._update_fluor_list()
            # Reselect the item
            self.fluor_tree.selection_set(f'fluor_{idx}')
            
    def get_fluorophores(self):
        """Get list of Fluorophore objects."""
        return [
            Fluorophore(
                name=data['name'],
                model=data['model'],
                params=data['params'],
                brightness=data['brightness']
            )
            for data in self.fluor_data
        ]
