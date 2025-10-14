"""Object property editing logic."""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Any, List
import tkinter as tk

if TYPE_CHECKING:
    from .manager import ObjectLayersManager


class ObjectEditor:
    """Handles editing of individual object properties."""
    
    def __init__(self, manager: 'ObjectLayersManager'):
        """Initialize the object editor.
        
        Args:
            manager: Reference to the parent ObjectLayersManager
        """
        self.manager = manager
    
    def auto_save(self) -> None:
        """Save current editor values back to the selected object.
        
        This is called whenever any property is changed via the UI.
        Prevents infinite loops via recursion guards.
        """
        sel = self.manager.obj_tree.selection()
        if not sel:
            return
        
        idx = self.manager.obj_tree.index(sel[0])
        if not (0 <= idx < len(self.manager.objects)):
            return
        
        try:
            # Build composition based on object mode
            mode = self.manager.obj_mode.get()
            composition = []
            binary_fluor = None
            
            if mode == "single":
                # Single-fluorophore mode: read from obj_fluor dropdown
                from .utils import fluorophore_name_to_index, get_fluorophore_list
                fluor_list = get_fluorophore_list(self.manager.get_fluorophore_names)
                fluor_name = self.manager.obj_fluor.get()
                if fluor_name and fluor_list:
                    fluor_idx = fluorophore_name_to_index(fluor_name, fluor_list)
                    composition.append({
                        'fluor_index': fluor_idx,
                        'ratio': 1.0
                    })
            else:
                # Multi-fluorophore mode: read from ratio sliders and binary dropdown
                # Add continuous fluorophores from ratio sliders
                if hasattr(self.manager, 'ratio_vars'):
                    # ratio_vars[i] corresponds to continuous_fluor_indices[i]
                    for i, var in enumerate(self.manager.ratio_vars):
                        if i < len(self.manager.continuous_fluor_indices):
                            fluor_idx = self.manager.continuous_fluor_indices[i]
                            ratio = var.get()
                            if ratio > 0:
                                composition.append({
                                    'fluor_index': fluor_idx,
                                    'ratio': float(ratio)
                                })
                
                # Add binary fluorophore
                if hasattr(self.manager, 'binary_fluor_var'):
                    from .utils import fluorophore_name_to_index, get_fluorophore_list
                    binary_name = self.manager.binary_fluor_var.get()
                    if binary_name:
                        fluor_list = get_fluorophore_list(self.manager.get_fluorophore_names)
                        binary_fluor = fluorophore_name_to_index(binary_name, fluor_list)
                        composition.append({
                            'fluor_index': binary_fluor,
                            'ratio': 1.0
                        })
            
            # Build region
            region_type = self.manager.obj_region_type.get()
            region = {'type': region_type}
            if region_type == 'rect':
                region.update({'x0': self.manager.obj_x0.get(), 'y0': self.manager.obj_y0.get(), 
                              'w': self.manager.obj_w.get(), 'h': self.manager.obj_h.get()})
            elif region_type == 'circle':
                region.update({'cx': self.manager.obj_cx.get(), 'cy': self.manager.obj_cy.get(), 
                              'r': self.manager.obj_r.get()})
            
            # Compute sigma from radius
            radius = self.manager.obj_radius.get()
            sigma = radius / 2.0
            
            # Update object
            obj_spec = {
                'name': self.manager.obj_name.get(),
                'mode': self.manager.obj_mode.get(),  # NEW: Save per-object mode
                'composition': composition,
                'binary_fluor': binary_fluor,
                'kind': self.manager.obj_kind.get(),
                'region': region,
                'count': self.manager.obj_count.get(),
                'spot_sigma': sigma,
                'radius': radius,
                'intensity_min': self.manager.obj_i_min.get(),
                'intensity_max': self.manager.obj_i_max.get(),
                'size_px': self.manager.obj_size.get(),
            }
            
            self.manager.objects[idx] = obj_spec
            self.manager._update_tree_item(idx)  # Update only this item, not full refresh
            
        except (ValueError, IndexError) as e:
            self.manager.log(f"Error saving object: {e}")
    
    def on_count_changed(self) -> None:
        """Handle count spinbox changes and sync with lambda."""
        if self.manager.count_lambda_sync_lock:
            return
        
        self.manager.count_lambda_sync_lock = True
        try:
            H, W = self.manager.get_image_dims()
            area_px = H * W
            count = self.manager.obj_count.get()
            lambda_val = count / area_px if area_px > 0 else 0.0
            self.manager.obj_lambda.set(lambda_val)
            
            if self.manager.lambda_display_label:
                self.manager.lambda_display_label.config(text=f'{lambda_val:.2e}')
            
            self.auto_save()
        finally:
            self.manager.count_lambda_sync_lock = False
    
    def on_lambda_changed(self) -> None:
        """Handle lambda spinbox changes and sync with count."""
        if self.manager.count_lambda_sync_lock:
            return
        
        self.manager.count_lambda_sync_lock = True
        try:
            H, W = self.manager.get_image_dims()
            area_px = H * W
            lambda_val = self.manager.obj_lambda.get()
            count = int(lambda_val * area_px)
            self.manager.obj_count.set(count)
            
            if self.manager.count_display_label:
                self.manager.count_display_label.config(text=str(count))
            
            self.auto_save()
        finally:
            self.manager.count_lambda_sync_lock = False
    
    def on_region_type_change(self, event=None) -> None:
        """Handle region type combobox selection."""
        region_type = self.manager.obj_region_type.get()
        
        # Clear old region params UI
        if self.manager.region_params_frame:
            for widget in self.manager.region_params_frame.winfo_children():
                widget.destroy()
        
        # Show relevant region parameters
        if region_type == "rect":
            self.manager.log("Region set to rectangle - configure x0, y0, w, h")
        elif region_type == "circle":
            self.manager.log("Region set to circle - configure center and radius")
        
        self.auto_save()
    
    def sync_radius_to_sigma(self) -> None:
        """Sync radius variable to sigma (sigma = radius / 2.0).
        
        Uses recursion guard to prevent infinite loops.
        """
        # Prevent recursive calls
        if hasattr(self.manager, '_syncing_radius') and self.manager._syncing_radius:
            return
        self.manager._syncing_radius = True
        try:
            radius = self.manager.obj_radius.get()
            sigma = radius / 2.0
            self.manager.obj_sigma.set(sigma)
            self.auto_save()
        finally:
            self.manager._syncing_radius = False

