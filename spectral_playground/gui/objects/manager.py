"""Object layers manager for the spectral visualization GUI (REFACTORED)."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
import copy
from typing import Callable, List, Dict, Any, Tuple

from .presets import PresetGenerator
from .ui_components import ObjectLayersUI
from .object_editor import ObjectEditor
from .composition_editor import CompositionEditor
from .utils import get_fluorophore_list, fluorophore_name_to_index, fluorophore_index_to_name


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
        
        # Debouncing timers to prevent excessive updates
        self._save_timer = None
        self._normalizing = False
        self._syncing_radius = False
        
        # Global composition settings
        self.object_mode = tk.StringVar(value="single")  # "single" or "multi" - for NEW objects
        self.num_continuous_fluors = tk.IntVar(value=2)
        self.continuous_fluor_indices = [0, 1]  # Explicit list of continuous fluorophore indices
        self.use_dirichlet = tk.BooleanVar(value=True)
        self.next_object_number = 1  # Auto-increment for object names
        
        # Object editor variables (per-object)
        self.obj_name = tk.StringVar(value="O1")
        self.obj_mode = tk.StringVar(value="single")  # Per-object mode selector
        self.obj_fluor = tk.StringVar(value="F1")
        self.obj_kind = tk.StringVar(value="gaussian_blobs")
        self.obj_count = tk.IntVar(value=50)
        self.obj_size = tk.DoubleVar(value=6.0)
        self.obj_i_min = tk.DoubleVar(value=0.5)
        self.obj_i_max = tk.DoubleVar(value=1.5)
        self.obj_sigma = tk.DoubleVar(value=2.0)
        self.obj_radius = tk.DoubleVar(value=4.0)
        self.obj_region_type = tk.StringVar(value="full")
        self.obj_x0 = tk.IntVar(value=0)
        self.obj_y0 = tk.IntVar(value=0)
        self.obj_w = tk.IntVar(value=64)
        self.obj_h = tk.IntVar(value=64)
        self.obj_cx = tk.DoubleVar(value=64)
        self.obj_cy = tk.DoubleVar(value=64)
        self.obj_r = tk.DoubleVar(value=40)
        
        # Statistical analysis parameters
        self.obj_lambda = tk.DoubleVar(value=0.0)
        self.count_lambda_sync_lock = False
        
        # Composition editor variables
        self.ratio_vars: List[tk.DoubleVar] = []
        self.binary_fluor_var = tk.StringVar(value="")
        self.binary_combo = None  # Reference to binary dropdown for dynamic updates
        
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
        
        # Initialize sub-components
        self.object_editor = ObjectEditor(self)
        self.composition_editor = CompositionEditor(self)
        
        # Build UI and add presets
        ObjectLayersUI.build_main_ui(self)
        self._add_preset_objects()
        
    # ===== Object Management Methods =====
    
    def _add_object(self) -> None:
        """Add a new object with auto-generated name and composition."""
        num_fluors = len(self.get_fluorophore_names())
        if num_fluors == 0:
            self.log("No fluorophores available")
            return
        
        # Use current image dimensions for sensible defaults
        H, W = self.get_image_dims()
        sigma = max(1.5, min(W, H) / 40)
        radius = 2.0 * sigma
        
        # Create object based on global mode (for new objects)
        if self.object_mode.get() == "single":
            # Classic single-fluorophore object
            obj = {
                'name': f'O{self.next_object_number}',
                'mode': 'single',  # NEW: per-object mode
                'composition': [{'fluor_index': 0, 'ratio': 1.0}],
                'binary_fluor': None,
                'kind': 'gaussian_blobs',
                'region': {'type': 'full'},
                'count': 50,
                'spot_sigma': sigma,
                'radius': radius,
                'intensity_min': 0.5,
                'intensity_max': 1.5,
                'size_px': max(3.0, min(W, H) / 20),
            }
        else:
            # Multi-fluorophore object
            from .composition import CompositionGenerator
            
            # Compute binary indices (all fluorophores NOT in continuous list)
            binary_indices = [i for i in range(num_fluors) if i not in self.continuous_fluor_indices]
            
            comp_data = CompositionGenerator.generate_composition(
                num_total_fluors=num_fluors,
                continuous_indices=self.continuous_fluor_indices,
                binary_indices=binary_indices,
                use_dirichlet=self.use_dirichlet.get()
            )
            
            obj = {
                'name': f'O{self.next_object_number}',
                'mode': 'multi',  # NEW: per-object mode
                'composition': comp_data['composition'],
                'binary_fluor': comp_data['binary_fluor'],
                'kind': 'gaussian_blobs',
                'region': {'type': 'full'},
                'count': 50,
                'spot_sigma': sigma,
                'radius': radius,
                'intensity_min': 0.5,
                'intensity_max': 1.5,
                'size_px': max(3.0, min(W, H) / 20),
            }
        
        self.objects.append(obj)
        self.next_object_number += 1
        self._refresh_object_list()
        
        # Auto-select the new object
        items = self.obj_tree.get_children()
        if items:
            self.obj_tree.selection_set(items[-1])
            self._on_object_select()
        
        self.log(f"Added {obj['name']}")

    def _remove_object(self) -> None:
        """Remove selected object from the list."""
        sel = self.obj_tree.selection()
        if not sel:
            return
        idx = self.obj_tree.index(sel[0])
        if 0 <= idx < len(self.objects):
            removed = self.objects.pop(idx)
            self._refresh_object_list()
            self.log(f"Removed object {removed.get('name', 'Unknown')}")

    def _duplicate_object(self) -> None:
        """Duplicate the selected object."""
        sel = self.obj_tree.selection()
        if not sel:
            return
        idx = self.obj_tree.index(sel[0])
        if 0 <= idx < len(self.objects):
            new_obj = copy.deepcopy(self.objects[idx])
            new_obj['name'] = f'O{self.next_object_number}'
            self.next_object_number += 1
            self.objects.append(new_obj)
            self._refresh_object_list()
            self.log(f"Duplicated object to {new_obj['name']}")

    def _refresh_object_list(self) -> None:
        """Refresh the object tree view."""
        from .composition import CompositionGenerator
        
        # Clear tree
        for item in self.obj_tree.get_children():
            self.obj_tree.delete(item)
        
        # Repopulate
        for idx, obj in enumerate(self.objects):
            name = obj.get('name', f'O{idx+1}')
            count = obj.get('count', 0)
            
            # Get radius (prefer 'radius', fall back to spot_sigma conversion)
            radius = obj.get('radius', obj.get('spot_sigma', 2.0) * 2.0)
            
            # Format composition display
            composition = obj.get('composition', [])
            binary_fluor = obj.get('binary_fluor')
            fluor_names = self.get_fluorophore_names()
            comp_display = CompositionGenerator.composition_to_display(composition, binary_fluor, fluor_names)
            
            self.obj_tree.insert('', 'end', values=(
                name,
                comp_display,
                count,
                f"{radius:.1f}px"
            ))
        
    def _update_tree_item(self, idx: int) -> None:
        """Update a single item in the tree (more efficient than full refresh).
        
        Args:
            idx: Index of the object to update
        """
        from .composition import CompositionGenerator
        
        if not (0 <= idx < len(self.objects)):
            return
        
        obj = self.objects[idx]
        name = obj.get('name', f'O{idx+1}')
        count = obj.get('count', 0)
        radius = obj.get('radius', obj.get('spot_sigma', 2.0) * 2.0)
        
        composition = obj.get('composition', [])
        binary_fluor = obj.get('binary_fluor')
        fluor_names = self.get_fluorophore_names()
        comp_display = CompositionGenerator.composition_to_display(composition, binary_fluor, fluor_names)
        
        # Get tree item ID
        items = self.obj_tree.get_children()
        if idx < len(items):
            self.obj_tree.item(items[idx], values=(
                name,
                comp_display,
                count,
                f"{radius:.1f}px"
            ))
        
    def _on_object_select(self, event=None) -> None:
        """Handle object selection."""
        sel = self.obj_tree.selection()
        if not sel:
            return
        idx = self.obj_tree.index(sel[0])
        if not (0 <= idx < len(self.objects)):
            return
        obj = self.objects[idx]
        
        # Backward compatibility: infer mode if not present
        obj_mode = obj.get('mode')
        if obj_mode is None:
            # Infer from composition
            comp = obj.get('composition', [])
            obj_mode = 'single' if len(comp) == 1 and comp[0].get('ratio', 1.0) == 1.0 else 'multi'
            obj['mode'] = obj_mode  # Save for future
        
        # Set per-object mode and rebuild UI
        self.obj_mode.set(obj_mode)
        self._on_mode_change(trigger_save=False)  # Rebuild UI without saving (we're loading)
        
        # Update name
        self.obj_name.set(str(obj.get('name', f'O{idx+1}')))
        
        # Update composition - load ratios and binary (for multi mode)
        composition = obj.get('composition', [])
        binary_fluor_idx = obj.get('binary_fluor')
        
        # Set ratio sliders for continuous fluorophores
        if hasattr(self, 'ratio_vars'):
            # ratio_vars[i] corresponds to continuous_fluor_indices[i]
            for i, fluor_idx in enumerate(self.continuous_fluor_indices):
                if i < len(self.ratio_vars):
                    # Find this fluorophore in composition
                    ratio = next((c['ratio'] for c in composition if c['fluor_index'] == fluor_idx), 0.0)
                    self.ratio_vars[i].set(ratio)
        
        # Set binary dropdown
        if binary_fluor_idx is not None and hasattr(self, 'binary_fluor_var'):
            fluor_names = self.get_fluorophore_names()
            if binary_fluor_idx < len(fluor_names):
                self.binary_fluor_var.set(fluor_names[binary_fluor_idx])
        
        # Update other properties
        self.obj_kind.set(str(obj.get('kind', 'gaussian_blobs')))
        self.obj_count.set(int(obj.get('count', 50)))
        
        # Update radius (and auto-sync sigma)
        radius = obj.get('radius', obj.get('spot_sigma', 2.0) * 2.0)
        self.obj_radius.set(float(radius))
        
        self.obj_i_min.set(float(obj.get('intensity_min', 0.5)))
        self.obj_i_max.set(float(obj.get('intensity_max', 1.5)))
        
        # Update region
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
    
    def _on_mode_change(self, trigger_save=True) -> None:
        """Handle per-object mode change - rebuild composition UI.
        
        Args:
            trigger_save: Whether to trigger auto_save after rebuilding UI
        """
        # Clear composition editor frame
        if hasattr(self, 'comp_editor_frame'):
            for widget in self.comp_editor_frame.winfo_children():
                widget.destroy()
        
        mode = self.obj_mode.get()
        
        if mode == "single":
            # Show simple fluorophore dropdown
            self._build_single_fluor_selector()
        else:
            # Show multi-fluorophore composition editor
            self.composition_editor.build(self.comp_editor_frame)
        
        # Trigger save to update object (only if requested)
        if trigger_save:
            self.object_editor.auto_save()
    
    def _build_single_fluor_selector(self) -> None:
        """Build simple fluorophore dropdown for single-fluorophore mode."""
        if not hasattr(self, 'comp_editor_frame'):
            return
        
        fluor_names = self.get_fluorophore_names()
        if not fluor_names:
            ttk.Label(self.comp_editor_frame, text="No fluorophores loaded").pack(padx=4, pady=4)
            return
        
        frame = ttk.Frame(self.comp_editor_frame)
        frame.pack(fill=tk.X, padx=4, pady=4)
        
        ttk.Label(frame, text="Fluorophore:", width=10).pack(side=tk.LEFT)
        
        # Get current single fluorophore (if any)
        sel = self.obj_tree.selection()
        current_fluor = "F1"
        if sel:
            idx = self.obj_tree.index(sel[0])
            if 0 <= idx < len(self.objects):
                obj = self.objects[idx]
                comp = obj.get('composition', [])
                if comp and len(comp) == 1:
                    fluor_idx = comp[0].get('fluor_index', 0)
                    if fluor_idx < len(fluor_names):
                        current_fluor = fluor_names[fluor_idx]
        
        self.obj_fluor.set(current_fluor)
        
        fluor_combo = ttk.Combobox(frame, textvariable=self.obj_fluor, 
                                   values=fluor_names, state='readonly', width=15)
        fluor_combo.pack(side=tk.LEFT, padx=4)
        fluor_combo.bind('<<ComboboxSelected>>', lambda e: self._on_single_fluor_change())
    
    def _on_single_fluor_change(self) -> None:
        """Handle single fluorophore selection change."""
        from .utils import fluorophore_name_to_index, get_fluorophore_list
        
        fluor_list = get_fluorophore_list(self.get_fluorophore_names)
        fluor_name = self.obj_fluor.get()
        fluor_idx = fluorophore_name_to_index(fluor_name, fluor_list)
        
        # Update composition to single fluorophore
        sel = self.obj_tree.selection()
        if sel:
            idx = self.obj_tree.index(sel[0])
            if 0 <= idx < len(self.objects):
                self.objects[idx]['composition'] = [{'fluor_index': fluor_idx, 'ratio': 1.0}]
                self.objects[idx]['binary_fluor'] = None
                self._update_tree_item(idx)
    
    # ===== Preset/Quick-Assign Methods =====
    
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
    
    def _quick_assign_single(self) -> None:
        """Generate single-fluorophore objects (one per fluorophore)."""
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
        
        num_fluors = len(fluorophore_names)
        
        # Generate objects using PresetGenerator
        objects, log_msg = PresetGenerator.quick_assign_single_fluorophore(
            num_fluors=num_fluors,
            img_h=img_h,
            img_w=img_w
        )
        
        # Clear existing objects and add new ones
        self.objects.clear()
        self.objects.extend(objects)
        
        # Update next object number
        if objects:
            max_num = max(int(obj['name'][1:]) for obj in objects if obj.get('name', '').startswith('O'))
            self.next_object_number = max_num + 1
        
        # Refresh the list
        self._refresh_object_list()
        
        # Log summary
        for line in log_msg.split('\n'):
            self.log(line)
    
    def _quick_assign_multi(self) -> None:
        """Generate multi-fluorophore objects (one per binary fluorophore)."""
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
        
        num_fluors = len(fluorophore_names)
        
        # Compute binary indices (all fluorophores NOT in continuous list)
        binary_indices = [i for i in range(num_fluors) if i not in self.continuous_fluor_indices]
        
        # Generate objects using PresetGenerator (new multi-fluorophore method)
        objects, log_msg = PresetGenerator.quick_assign_multi_fluorophore(
            num_fluors=num_fluors,
            continuous_indices=self.continuous_fluor_indices,
            binary_indices=binary_indices,
            use_dirichlet=self.use_dirichlet.get(),
            img_h=img_h,
            img_w=img_w
        )
        
        # Clear existing objects and add new ones
        self.objects.clear()
        self.objects.extend(objects)
        
        # Update next object number
        if objects:
            max_num = max(int(obj['name'][1:]) for obj in objects if obj.get('name', '').startswith('O'))
            self.next_object_number = max_num + 1
        
        # Refresh the list
        self._refresh_object_list()
        
        # Log summary
        for line in log_msg.split('\n'):
            self.log(line)
    
    # ===== Public API =====
    
    def get_objects(self) -> List[Dict[str, Any]]:
        """Get list of object specifications."""
        converted_objects = []
        for obj in self.objects:
            obj_copy = copy.deepcopy(obj)
            
            # Ensure spot_sigma is set (derive from radius if needed)
            if 'radius' in obj_copy and 'spot_sigma' not in obj_copy:
                obj_copy['spot_sigma'] = obj_copy['radius'] / 2.0
            
            converted_objects.append(obj_copy)
        
        return converted_objects

    def should_include_base_field(self) -> bool:
        """Check if base field should be included in generation."""
        return self.include_base_field.get()
    
    def _open_fluorophore_config(self):
        """Open dialog to configure continuous vs binary fluorophore roles."""
        dialog = tk.Toplevel(self.parent_frame)
        dialog.title("Configure Fluorophore Roles")
        dialog.geometry("400x300")
        
        # Get available fluorophores
        fluor_names = self.get_fluorophore_names()
        if not fluor_names:
            ttk.Label(dialog, text="No fluorophores loaded").pack(pady=20)
            ttk.Button(dialog, text="Close", command=dialog.destroy).pack()
            return
        
        # Instructions
        ttk.Label(dialog, text="Select continuous (ratio) fluorophores:", 
                 font=('TkDefaultFont', 10, 'bold')).pack(pady=(10, 5))
        ttk.Label(dialog, text="Unselected fluorophores will be binary (one-of).", 
                 foreground='gray').pack(pady=(0, 10))
        
        # Checkboxes for each fluorophore
        continuous_vars = []
        checkbox_frame = ttk.Frame(dialog)
        checkbox_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        for i, fluor_name in enumerate(fluor_names):
            var = tk.BooleanVar(value=(i in self.continuous_fluor_indices))
            continuous_vars.append(var)
            ttk.Checkbutton(checkbox_frame, text=fluor_name, variable=var).pack(anchor='w', pady=2)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        
        def save_config():
            continuous_indices = [i for i, var in enumerate(continuous_vars) if var.get()]
            self.continuous_fluor_indices = continuous_indices if continuous_indices else [0]
            self.num_continuous_fluors.set(len(self.continuous_fluor_indices))
            
            # Update label
            if hasattr(self, 'fluor_config_label'):
                cont_names = [fluor_names[i] for i in self.continuous_fluor_indices if i < len(fluor_names)]
                binary_count = len(fluor_names) - len(self.continuous_fluor_indices)
                self.fluor_config_label.config(
                    text=f"{','.join(cont_names[:2])}{'...' if len(cont_names)>2 else ''} (ratio) | {binary_count} binary"
                )
            
            dialog.destroy()
        
        ttk.Button(button_frame, text="Save", command=save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

