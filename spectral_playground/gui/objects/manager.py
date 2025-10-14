"""Object layers manager for the spectral visualization GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox
import copy
from typing import Callable, List, Dict, Any, Tuple

from .templates import TemplateManager
from .presets import PresetGenerator
from .ui_components import ObjectLayersUI
from .dialogs import TemplateEditorDialog


class ObjectLayersManager:
    """Manager for object layers that can be placed on the image."""
    
    def __init__(
        self,
        parent_frame: tk.Widget,
        log_callback: Callable[[str], None],
        get_image_dims_callback: Callable[[], Tuple[int, int]],
        get_fluorophore_names_callback: Callable[[], List[str]]
    ):
        """Initialize the object layers manager.
        
        Args:
            parent_frame: Parent tkinter frame
            log_callback: Function to log messages
            get_image_dims_callback: Function to get image dimensions (H, W)
            get_fluorophore_names_callback: Function to get list of fluorophore names
        """
        self.parent_frame = parent_frame
        self.log = log_callback
        self.get_image_dims = get_image_dims_callback
        self.get_fluorophore_names = get_fluorophore_names_callback
        self.objects: List[Dict[str, Any]] = []
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
        
        # Statistical analysis parameters
        self.obj_lambda = tk.DoubleVar(value=0.0)  # Spatial intensity (objects/px²)
        self.count_lambda_sync_lock = False  # Prevent update loops
        
        # Composition mode: 'single' or 'template'
        self.composition_mode = tk.StringVar(value="single")
        self.composition_template = tk.StringVar(value="F1 Only")
        
        # UI widgets (will be set by UI builder)
        self.obj_tree = None
        self.fluor_selection_frame = None
        self.single_fluor_frame = None
        self.template_frame = None
        self.fluor_combo = None
        self.template_combo = None
        self.kind_combo = None
        self.count_display_label = None
        self.lambda_display_label = None
        self.size_label = None
        self.size_entry = None
        self.sigma_label = None
        self.sigma_entry = None
        self.region_params_frame = None
        
        # Build UI and add presets
        ObjectLayersUI.build_main_ui(self)
        self._add_preset_objects()
        
    # ===== Object Management Methods =====
    
    def _add_object(self, template_name: str | None = None) -> None:
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

    def _remove_object(self) -> None:
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

    def _duplicate_object(self) -> None:
        """Duplicate selected object."""
        sel = self.obj_tree.selection()
        if not sel:
            return
        idx = self.obj_tree.index(sel[0])
        if 0 <= idx < len(self.objects):
            self.objects.append(copy.deepcopy(self.objects[idx]))
            self._refresh_object_list()

    def _refresh_object_list(self) -> None:
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
        
    def _on_object_select(self, event=None) -> None:
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
        
        # Initialize lambda from count
        H, W = self.get_image_dims()
        area = H * W
        if area > 0:
            lambda_val = self.obj_count.get() / area
            self.obj_lambda.set(lambda_val)
            self.count_display_label.config(text=f"→ λ = {lambda_val:.6f} obj/px²")
            self.lambda_display_label.config(text=f"→ n ≈ {self.obj_count.get()} objects")
        
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
        ObjectLayersUI.update_region_ui(self)
        
        # Update kind-specific UI elements
        self._on_kind_change()

    def _auto_save(self) -> None:
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
                obj_spec = {
                    'template_name': template_name,
                    'kind': self.obj_kind.get(),
                    'region': region,
                    'count': int(self.obj_count.get()),
                    'size_px': float(self.obj_size.get()),
                    'intensity_min': float(self.obj_i_min.get()),
                    'intensity_max': float(self.obj_i_max.get()),
                    'spot_sigma': float(self.obj_sigma.get()),
                }
                self.objects[idx] = obj_spec
            else:
                # Single fluorophore mode
                fluor_name = self.obj_fluor.get()
                fluor_index = self._fluorophore_name_to_index(fluor_name)
                
                obj_spec = {
                    'fluor_index': fluor_index,
                    'kind': self.obj_kind.get(),
                    'region': region,
                    'count': int(self.obj_count.get()),
                    'size_px': float(self.obj_size.get()),
                    'intensity_min': float(self.obj_i_min.get()),
                    'intensity_max': float(self.obj_i_max.get()),
                    'spot_sigma': float(self.obj_sigma.get()),
                }
                self.objects[idx] = obj_spec
            self._refresh_object_list()
            
            # Re-select the updated item
            items = self.obj_tree.get_children()
            if idx < len(items):
                self.obj_tree.selection_set(items[idx])
            
        except Exception:
            pass  # Silently fail on invalid input during typing
    
    # ===== Event Handlers =====
    
    def _on_count_changed(self) -> None:
        """Handle count field change - update lambda."""
        if self.count_lambda_sync_lock:
            return
        self.count_lambda_sync_lock = True
        
        try:
            H, W = self.get_image_dims()
            area = H * W
            count = self.obj_count.get()
            lambda_val = count / area if area > 0 else 0.0
            
            self.obj_lambda.set(lambda_val)
            self.count_display_label.config(text=f"→ λ = {lambda_val:.6f} obj/px²")
        except:
            pass
        finally:
            self.count_lambda_sync_lock = False
            self._auto_save()
    
    def _on_lambda_changed(self) -> None:
        """Handle lambda field change - update count."""
        if self.count_lambda_sync_lock:
            return
        self.count_lambda_sync_lock = True
        
        try:
            H, W = self.get_image_dims()
            area = H * W
            lambda_val = self.obj_lambda.get()
            count = int(round(lambda_val * area))
            
            self.obj_count.set(count)
            self.lambda_display_label.config(text=f"→ n ≈ {count} objects")
        except:
            pass
        finally:
            self.count_lambda_sync_lock = False
            self._auto_save()
    
    def _on_kind_change(self, event=None) -> None:
        """Handle object kind change to show/hide relevant parameters."""
        kind = self.obj_kind.get()
        
        # Show/hide parameters based on object kind
        if kind in ("gaussian_blobs", "dots"):
            # Gaussian blobs and dots: ONLY use spot_sigma (size_px is ignored in code!)
            self.size_label.grid_remove()
            self.size_entry.grid_remove()
            
            # Show sigma controls
            self.sigma_label.grid(row=6, column=0, sticky='w', pady=(6,0))
            self.sigma_entry.grid(row=6, column=1, columnspan=3, sticky='w', pady=(6,0))
        else:
            # Circles and boxes: use size_px for radius/dimensions, no sigma
            self.size_label.grid(row=4, column=0, sticky='w', padx=(0,4), pady=(6,0))
            self.size_entry.grid(row=4, column=1, sticky='w', pady=(6,0))
            self.sigma_label.grid_remove()
            self.sigma_entry.grid_remove()
    
    def _on_composition_mode_change(self) -> None:
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
    
    def _on_region_type_change(self, event=None) -> None:
        """Handle region type change."""
        # Update UI to show only relevant parameters
        ObjectLayersUI.update_region_ui(self)
        
        # Visual feedback when changing region type
        region_type = self.obj_region_type.get()
        if region_type == "full":
            self.log("Region set to full image")
        elif region_type == "rect":
            self.log("Region set to rectangle - configure x0, y0, width, height")
        elif region_type == "circle":
            self.log("Region set to circle - configure center and radius")
    
    def _open_template_manager(self) -> None:
        """Open template manager dialog."""
        dialog = TemplateEditorDialog(
            self.parent_frame,
            self.template_manager,
            self._fluorophore_index_to_name,
            self._fluorophore_name_to_index,
            self._get_fluorophore_list,
            self.log
        )
        dialog.show()
    
    # ===== Preset Generation =====
    
    def _add_preset_objects(self) -> None:
        """Add 3 preset objects, one for each default fluorophore."""
        H, W = self.get_image_dims()
        presets = PresetGenerator.generate_default_presets(H, W)
        
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
    
    def _quick_assign_sample(self) -> None:
        """Generate a sample dataset with all available fluorophores."""
        # Get image dimensions
        try:
            img_h, img_w = self.get_image_dims()
        except:
            self.log("Cannot determine image size. Using default 128x128.")
            img_h, img_w = 128, 128
        
        # Get available fluorophores
        fluorophore_names = self.get_fluorophore_names()
        if not fluorophore_names:
            messagebox.showwarning("No Fluorophores", "Please add fluorophores first!")
            return
        
        # Generate objects using PresetGenerator
        objects, log_msg = PresetGenerator.generate_quick_assign_sample(
            img_h, img_w, fluorophore_names
        )
        
        # Clear existing objects and add new ones
        self.objects.clear()
        self.objects.extend(objects)
        
        # Refresh the list
        self._refresh_object_list()
        
        # Log summary
        for line in log_msg.split('\n'):
            self.log(line)
    
    # ===== Public API =====
    
    def get_objects(self) -> List[Dict[str, Any]]:
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
        
    def should_include_base_field(self) -> bool:
        """Check if base field should be included."""
        return self.include_base_field.get()
    
    # ===== Helper Methods =====
    
    def _get_fluorophore_list(self) -> List[str]:
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

