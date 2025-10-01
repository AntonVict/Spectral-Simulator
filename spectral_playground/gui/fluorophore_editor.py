"""Fluorophore editor component for the spectral visualization GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import csv
from pathlib import Path

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
        # Name field
        name_frame = ttk.Frame(self)
        name_frame.pack(fill=tk.X, pady=2)
        ttk.Label(name_frame, text="Name:").grid(row=0, column=0, sticky="w", padx=(0,4))
        self.name_var = tk.StringVar(value=self.data['name'])
        name_entry = ttk.Entry(name_frame, textvariable=self.name_var, width=15)
        name_entry.grid(row=0, column=1, padx=2)
        # Save on Enter key or when focus leaves the field (not on every keystroke)
        name_entry.bind('<Return>', self._on_name_change)
        name_entry.bind('<FocusOut>', self._on_name_change)
        
        # Model selection
        model_frame = ttk.Frame(self)
        model_frame.pack(fill=tk.X, pady=2)
        ttk.Label(model_frame, text="Model:").grid(row=0, column=0, sticky="w", padx=(0,4))
        self.model_var = tk.StringVar(value=self.data['model'])
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                 values=["gaussian", "skewnorm", "lognormal", "weibull", "empirical"], 
                                 width=12, state="readonly")
        self.model_combo.grid(row=0, column=1, padx=2)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)
        
        # Disable model selection for empirical (imported) fluorophores
        if self.data['model'] == 'empirical':
            self.model_combo.state(['disabled'])
        
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
        self._auto_save()
    
    def _on_name_change(self, event=None):
        """Handle name change."""
        self._auto_save()
        
    def _on_param_change(self, event=None):
        """Handle parameter value change."""
        self._auto_save()
    
    def _auto_save(self):
        """Automatically save changes to internal data."""
        self.data['name'] = self.name_var.get()
        self.data['model'] = self.model_var.get()
        try:
            self.data['brightness'] = self.brightness_var.get()
        except:
            pass
        
        # For empirical models, preserve existing params (CSV data)
        # For other models, rebuild from param_vars
        if self.model_var.get() == 'empirical':
            # Don't touch params - they contain the CSV data (csv_wavelengths, csv_intensities)
            # This prevents clearing the imported data when changing the name
            pass
        else:
            # Rebuild params from UI fields
            self.data['params'] = {}
            for k, v in self.param_vars.items():
                try:
                    self.data['params'][k] = v.get()
                except:
                    pass
        
        if self.on_update:
            self.on_update(self.fluor_idx, self.data)
            
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
        elif model == "empirical":
            # No editable parameters for imported signatures
            pass
            
    def _add_param(self, label, key, default_val):
        """Add a parameter input field."""
        row = len(self.param_vars)
        ttk.Label(self.params_frame, text=label).grid(row=row, column=0, sticky="w", padx=(0,4))
        var = tk.DoubleVar(value=default_val)
        self.param_vars[key] = var
        entry = ttk.Entry(self.params_frame, textvariable=var, width=8)
        entry.grid(row=row, column=1, padx=2, pady=1)
        entry.bind('<KeyRelease>', self._on_param_change)
        
        
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
        
        # Left side: Add/Remove buttons
        btn_frame = ttk.Frame(list_header)
        btn_frame.pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="+ Add", command=self.add_fluorophore, width=8).pack(side=tk.LEFT, padx=(0,2))
        ttk.Button(btn_frame, text="- Remove", command=self.remove_fluorophore, width=8).pack(side=tk.LEFT)
        
        # Right side: Import button
        ttk.Button(list_header, text="Import Signature(s)", command=self.import_signatures, width=18).pack(side=tk.RIGHT)
        
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
        
        # Editor panel for selected fluorophore - no bold label
        self.fluor_editor_frame = ttk.Frame(self.parent_frame, relief='groove', borderwidth=2)
        self.fluor_editor_frame.pack(fill=tk.X, pady=(8,2))
        
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
                # Only renumber auto-generated FX names (preserve imported names)
                auto_count = 0
                for i, data in enumerate(self.fluor_data):
                    # Check if this is an auto-generated name (F1, F2, F3, etc.)
                    if data['name'].startswith('F') and data['name'][1:].isdigit():
                        auto_count += 1
                        data['name'] = f'F{auto_count}'
                    # Otherwise keep the custom name (e.g., "FITC", "AF_750")
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
        elif model == 'empirical':
            return "Imported"
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
            
            
    def _on_fluor_update(self, idx, data):
        """Handle fluorophore data update."""
        # Update the stored data
        if idx < len(self.fluor_data):
            self.fluor_data[idx] = data
            self._update_fluor_list()
            # Reselect the item
            self.fluor_tree.selection_set(f'fluor_{idx}')
            
    def get_fluorophores(self):
        """Get list of Fluorophore objects.
        
        For empirical fluorophores, passes through raw CSV data which will be
        interpolated by SpectralSystem._pdf() to the correct wavelength grid.
        """
        fluorophores = []
        for data in self.fluor_data:
            params = data['params'].copy()
            
            # For empirical model: just pass through the params as-is
            # The SpectralSystem._pdf() will handle interpolation
            # No conversion needed here!
            
            fluorophores.append(Fluorophore(
                name=data['name'],
                model=data['model'],
                params=params,
                brightness=data['brightness']
            ))
        return fluorophores
    
    def set_fluorophores(self, fluorophores: list):
        """Set fluorophores from a list of Fluorophore objects."""
        self.fluor_data = []
        for fl in fluorophores:
            # Ensure params is a dictionary and can be copied safely
            if isinstance(fl.params, dict):
                params_copy = fl.params.copy()
            else:
                # Convert non-dict params to dict if needed
                params_copy = {'value': fl.params} if fl.params is not None else {}
                
            self.fluor_data.append({
                'name': fl.name,
                'model': fl.model,
                'brightness': fl.brightness,
                'params': params_copy
            })
        self._update_fluor_list()
        if self.fluor_data:
            # Select the first fluorophore by default
            first_item = f'fluor_0'
            if self.fluor_tree.exists(first_item):
                self.fluor_tree.selection_set(first_item)
                self._on_fluor_select()
    
    def import_signatures(self):
        """Import spectral signatures from CSV files."""
        filepaths = filedialog.askopenfilenames(
            title="Select Spectral Signature CSV Files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            parent=self.parent_frame
        )
        
        if not filepaths:
            return
        
        imported_count = 0
        errors = []
        
        for filepath in filepaths:
            try:
                name, raw_wavelengths, raw_intensities = self._parse_csv_signature(filepath)
                
                # Add as empirical fluorophore
                # Store raw CSV data for re-interpolation when wavelength grid changes
                fluor_data = {
                    'name': name,
                    'model': 'empirical',
                    'params': {
                        'csv_wavelengths': raw_wavelengths.tolist(),
                        'csv_intensities': raw_intensities.tolist()
                    },
                    'brightness': 1.0
                }
                
                self.fluor_data.append(fluor_data)
                imported_count += 1
                self.log(f"Imported: {name}")
                
            except Exception as e:
                errors.append(f"{Path(filepath).name}: {str(e)}")
        
        # Update the list
        if imported_count > 0:
            self._update_fluor_list()
            self.log(f"Successfully imported {imported_count} signature(s)")
            
            # Select the first imported item
            if self.fluor_data:
                last_idx = len(self.fluor_data) - imported_count
                self.fluor_tree.selection_set(f'fluor_{last_idx}')
                self._on_fluor_select()
        
        # Show errors if any
        if errors:
            error_msg = "Some files could not be imported:\n\n" + "\n".join(errors[:5])
            if len(errors) > 5:
                error_msg += f"\n\n... and {len(errors) - 5} more errors"
            messagebox.showwarning("Import Warnings", error_msg)
    
    def _get_wavelength_grid(self):
        """Get the current wavelength grid from the sidebar."""
        try:
            # Navigate up to find the sidebar
            parent = self.parent_frame
            while parent:
                if hasattr(parent, 'master'):
                    if hasattr(parent.master, 'panels') and 'grid' in parent.master.panels:
                        grid_config = parent.master.panels['grid'].get_wavelength_grid()
                        start = grid_config['start']
                        stop = grid_config['stop']
                        step = grid_config['step']
                        return np.arange(start, stop + step/2, step)
                    parent = parent.master
                else:
                    break
            return None
        except:
            return None
    
    def _parse_csv_signature(self, filepath):
        """Parse a CSV file containing spectral signature data.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            tuple: (name, wavelengths, intensities) - raw CSV data
            
        Raises:
            ValueError: If CSV format is invalid
        """
        filepath = Path(filepath)
        name = filepath.stem  # Filename without extension
        
        # Read CSV
        wavelengths = []
        intensities = []
        
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            
            # Check for required column
            if 'Emission_Normalized' not in reader.fieldnames:
                if 'Wavelength' not in reader.fieldnames:
                    raise ValueError("CSV must contain 'Wavelength' column")
                # Try other possible column names
                intensity_col = None
                for col in ['Emission_Normalized', 'Normalized', 'Intensity', 'PDF', 'Emission']:
                    if col in reader.fieldnames:
                        intensity_col = col
                        break
                if not intensity_col:
                    raise ValueError(f"CSV must contain emission data column (tried: Emission_Normalized, Normalized, Intensity)")
            else:
                intensity_col = 'Emission_Normalized'
            
            # Parse data
            for row in reader:
                try:
                    wl = float(row['Wavelength'])
                    intensity = float(row[intensity_col])
                    wavelengths.append(wl)
                    intensities.append(intensity)
                except (ValueError, KeyError) as e:
                    continue  # Skip invalid rows
        
        if len(wavelengths) == 0:
            raise ValueError("No valid data found in CSV")
        
        # Convert to numpy arrays
        wavelengths = np.array(wavelengths)
        intensities = np.array(intensities)
        
        # Validate wavelengths are monotonic
        if not np.all(np.diff(wavelengths) > 0):
            raise ValueError("Wavelengths must be monotonically increasing")
        
        # Ensure non-negative
        intensities = np.clip(intensities, 0.0, None)
        
        # Check for non-zero emission
        if np.sum(intensities) == 0:
            raise ValueError("Spectral signature has zero total emission")
        
        return name, wavelengths, intensities
