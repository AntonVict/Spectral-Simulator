"""Composition editor UI and logic for multi-fluorophore objects."""

from __future__ import annotations
from typing import TYPE_CHECKING, List
import tkinter as tk
from tkinter import ttk

if TYPE_CHECKING:
    from .manager import ObjectLayersManager


class CompositionEditor:
    """Handles the composition editor UI (ratio sliders + binary dropdown)."""
    
    def __init__(self, manager: 'ObjectLayersManager'):
        """Initialize the composition editor.
        
        Args:
            manager: Reference to the parent ObjectLayersManager
        """
        self.manager = manager
    
    def build(self, parent: ttk.Frame) -> None:
        """Build inline composition editor with sliders + binary dropdown.
        
        Args:
            parent: Parent frame to build the editor in
        """
        num_fluors = len(self.manager.get_fluorophore_names())
        fluor_names = self.manager.get_fluorophore_names()
        
        if num_fluors == 0:
            ttk.Label(parent, text="No fluorophores loaded").pack(padx=4, pady=4)
            return
        
        # Clear any existing ratio vars
        self.manager.ratio_vars = []
        
        # Continuous fluorophore ratio sliders (only for configured continuous fluorophores)
        for i in self.manager.continuous_fluor_indices:
            if i >= num_fluors:
                continue
            fluor_name = fluor_names[i]
            
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, padx=4, pady=2)
            
            ttk.Label(frame, text=f"{fluor_name}:", width=8).pack(side=tk.LEFT)
            
            # Equal initial ratio
            initial_ratio = 1.0 / len(self.manager.continuous_fluor_indices) if self.manager.continuous_fluor_indices else 1.0
            ratio_var = tk.DoubleVar(value=initial_ratio)
            self.manager.ratio_vars.append(ratio_var)
            
            # Don't use command= here - it causes infinite loops with var.set()
            scale = ttk.Scale(frame, from_=0.0, to=1.0, variable=ratio_var, 
                             orient=tk.HORIZONTAL)
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
            
            label = ttk.Label(frame, text=f"{ratio_var.get():.2f}", width=8)
            label.pack(side=tk.LEFT)
            
            # Update label and save when ratio changes
            def make_updater(lbl, var):
                def update(*args):
                    # Check if widget still exists before updating
                    try:
                        if lbl.winfo_exists():
                            lbl.config(text=f"{var.get():.2f}")
                    except tk.TclError:
                        # Widget was destroyed, stop updating
                        return
                    
                    # Use after_idle to debounce auto-save calls
                    if hasattr(self.manager, '_save_timer') and self.manager._save_timer:
                        self.manager.parent_frame.after_cancel(self.manager._save_timer)
                    self.manager._save_timer = self.manager.parent_frame.after(300, self.manager.object_editor.auto_save)  # 300ms delay
                return update
            ratio_var.trace('w', make_updater(label, ratio_var))
        
        # Binary fluorophore dropdown (only show non-continuous fluorophores)
        binary_indices = [i for i in range(num_fluors) if i not in self.manager.continuous_fluor_indices]
        if binary_indices:
            binary_frame = ttk.Frame(parent)
            binary_frame.pack(fill=tk.X, padx=4, pady=4)
            
            ttk.Label(binary_frame, text="Binary:", width=8).pack(side=tk.LEFT)
            
            binary_options = [fluor_names[i] for i in binary_indices]
            self.manager.binary_fluor_var.set(binary_options[0] if binary_options else "")
            
            # Store reference to combobox for dynamic updates
            self.manager.binary_combo = ttk.Combobox(binary_frame, textvariable=self.manager.binary_fluor_var, 
                                            values=binary_options, state='readonly', width=12)
            self.manager.binary_combo.pack(side=tk.LEFT, padx=4)
            self.manager.binary_fluor_var.trace('w', lambda *args: self.manager.object_editor.auto_save())
        
        # Randomize button
        ttk.Button(parent, text="Randomize Composition", 
                  command=self.randomize).pack(pady=4)
    
    def normalize_ratios(self) -> None:
        """Normalize continuous fluorophore ratios to sum to 1.0.
        
        Uses recursion guard to prevent cascading updates.
        """
        if hasattr(self.manager, '_normalizing') and self.manager._normalizing:
            return
        self.manager._normalizing = True
        try:
            if hasattr(self.manager, 'ratio_vars') and self.manager.ratio_vars:
                total = sum(var.get() for var in self.manager.ratio_vars)
                if total > 0:
                    for var in self.manager.ratio_vars:
                        var.set(var.get() / total)
            
            self.manager.object_editor.auto_save()
        finally:
            self.manager._normalizing = False
    
    def randomize(self) -> None:
        """Randomize the composition ratios and binary selection."""
        from .composition import CompositionGenerator
        
        num_fluors = len(self.manager.get_fluorophore_names())
        
        # Compute binary indices (all fluorophores NOT in continuous list)
        binary_indices = [i for i in range(num_fluors) if i not in self.manager.continuous_fluor_indices]
        
        comp_data = CompositionGenerator.generate_composition(
            num_total_fluors=num_fluors,
            continuous_indices=self.manager.continuous_fluor_indices,
            binary_indices=binary_indices,
            use_dirichlet=self.manager.use_dirichlet.get()
        )
        
        # Update ratio sliders (map fluorophore index to ratio_vars position)
        for comp in comp_data['composition']:
            fluor_idx = comp['fluor_index']
            # Find position in continuous_fluor_indices
            if fluor_idx in self.manager.continuous_fluor_indices:
                pos = self.manager.continuous_fluor_indices.index(fluor_idx)
                if pos < len(self.manager.ratio_vars):
                    self.manager.ratio_vars[pos].set(comp['ratio'])
        
        # Update binary dropdown
        if comp_data['binary_fluor'] is not None:
            fluor_names = self.manager.get_fluorophore_names()
            if comp_data['binary_fluor'] < len(fluor_names):
                self.manager.binary_fluor_var.set(fluor_names[comp_data['binary_fluor']])
        
        self.manager.object_editor.auto_save()
    
    def refresh_binary_dropdown(self) -> None:
        """Refresh the binary fluorophore dropdown with current fluorophore list."""
        if not hasattr(self.manager, 'binary_combo') or self.manager.binary_combo is None:
            return
        
        num_fluors = len(self.manager.get_fluorophore_names())
        fluor_names = self.manager.get_fluorophore_names()
        
        # Compute binary indices (all fluorophores NOT in continuous list)
        binary_indices = [i for i in range(num_fluors) if i not in self.manager.continuous_fluor_indices]
        
        if binary_indices:
            binary_options = [fluor_names[i] for i in binary_indices]
            
            # Update combobox values
            self.manager.binary_combo['values'] = binary_options
            
            # Keep current selection if it's still valid, otherwise select first option
            current = self.manager.binary_fluor_var.get()
            if current not in binary_options and binary_options:
                self.manager.binary_fluor_var.set(binary_options[0])

