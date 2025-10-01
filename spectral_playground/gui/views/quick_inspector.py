"""Quick Inspector Panel - compact object info in main window."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import List, Dict, Any, Callable, Optional
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class QuickInspectorPanel(ttk.Frame):
    """Compact inspector panel for quick object info in main window."""
    
    def __init__(
        self,
        parent: tk.Widget,
        get_data_callback: Callable,
        get_fluorophore_names_callback: Callable,
        on_open_full_inspector: Callable
    ):
        super().__init__(parent)
        self.get_data = get_data_callback
        self.get_fluorophore_names = get_fluorophore_names_callback
        self.on_open_full_inspector = on_open_full_inspector
        
        self.selected_object_ids: List[int] = []
        self.all_objects: List[Dict[str, Any]] = []
        
        self._build_ui()
    
    def _build_ui(self):
        """Build the quick inspector UI."""
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        
        # Header with selection count
        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky='ew', padx=4, pady=4)
        
        self.selection_label = ttk.Label(header, text="No objects selected", 
                                         font=('TkDefaultFont', 9, 'bold'))
        self.selection_label.pack(side=tk.LEFT)
        
        ttk.Button(header, text="Full Inspector >>", 
                  command=self.on_open_full_inspector, width=15).pack(side=tk.RIGHT)
        ttk.Button(header, text="Clear", 
                  command=self._clear_selection, width=8).pack(side=tk.RIGHT, padx=4)
        
        # Main content area (will show either mini view or object list)
        content_frame = ttk.Frame(self)
        content_frame.grid(row=1, column=0, sticky='nsew', padx=4, pady=4)
        content_frame.columnconfigure(0, weight=1)
        content_frame.rowconfigure(0, weight=1)
        
        # Create notebook for different views
        self.notebook = ttk.Notebook(content_frame)
        self.notebook.grid(row=0, column=0, sticky='nsew')
        
        # Tab 1: Object List
        list_frame = ttk.Frame(self.notebook)
        self.notebook.add(list_frame, text="Object List")
        self._build_object_list(list_frame)
        
        # Tab 2: Mini Composite View
        view_frame = ttk.Frame(self.notebook)
        self.notebook.add(view_frame, text="Zoomed View")
        self._build_mini_view(view_frame)
    
    def _build_object_list(self, parent: ttk.Frame):
        """Build compact object list."""
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        
        # Scrollable text widget for object info
        self.object_text = tk.Text(parent, height=10, wrap=tk.NONE, 
                                    font=('Courier', 9))
        self.object_text.grid(row=0, column=0, sticky='nsew')
        
        scrollbar = ttk.Scrollbar(parent, orient='vertical', 
                                 command=self.object_text.yview)
        self.object_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        # Initial message
        self._update_object_list()
    
    def _build_mini_view(self, parent: ttk.Frame):
        """Build mini composite view."""
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        
        # Info label
        self.view_info_label = ttk.Label(parent, 
            text="Select an object to see zoomed view", 
            foreground="gray")
        self.view_info_label.grid(row=0, column=0, pady=20)
        
        # Figure for mini composite
        self.mini_figure = Figure(figsize=(3, 3), dpi=80)
        self.mini_ax = self.mini_figure.add_subplot(111)
        self.mini_ax.set_xticks([])
        self.mini_ax.set_yticks([])
        
        self.mini_canvas = FigureCanvasTkAgg(self.mini_figure, parent)
        self.mini_canvas_widget = self.mini_canvas.get_tk_widget()
        # Don't grid it yet, show message first
    
    def set_selection(self, object_ids: List[int]):
        """Update the selected objects."""
        self.selected_object_ids = object_ids
        self._update_display()
    
    def refresh_objects(self):
        """Refresh object list from current dataset."""
        data = self.get_data()
        if not data or not data.has_data:
            self.all_objects = []
        else:
            self.all_objects = data.metadata.get('objects', [])
        
        self._update_display()
    
    def _update_display(self):
        """Update all display elements."""
        self._update_selection_label()
        self._update_object_list()
        self._update_mini_view()
    
    def _update_selection_label(self):
        """Update the selection count label."""
        if not self.selected_object_ids:
            self.selection_label.config(text="No objects selected")
        elif len(self.selected_object_ids) == 1:
            self.selection_label.config(text=f"Object #{self.selected_object_ids[0]} selected")
        else:
            self.selection_label.config(text=f"{len(self.selected_object_ids)} objects selected")
    
    def _update_object_list(self):
        """Update the object list text."""
        self.object_text.delete('1.0', tk.END)
        
        if not self.selected_object_ids:
            self.object_text.insert('1.0', "No objects selected.\n\n"
                                           "Click on objects in the Composite View\n"
                                           "or use the Full Inspector to select objects.")
            return
        
        fluor_names = self.get_fluorophore_names()
        
        # Header
        self.object_text.insert('1.0', f"{'ID':<6} {'Type':<14} {'Pos(y,x)':<14} {'Int':<8} {'Fluor'}\n")
        self.object_text.insert(tk.END, "-" * 70 + "\n")
        
        # Get selected object details
        selected_objs = [obj for obj in self.all_objects 
                        if obj['id'] in self.selected_object_ids]
        
        for obj in selected_objs:
            obj_id = obj['id']
            obj_type = obj.get('type', 'unknown')[:12]
            pos = obj.get('position', (0, 0))
            pos_str = f"({pos[0]:.1f},{pos[1]:.1f})"
            intensity = obj.get('base_intensity', 0.0)
            
            # Format composition
            comp = obj.get('composition', [])
            if len(comp) == 1:
                fluor_str = fluor_names[comp[0]['fluor_index']] if comp[0]['fluor_index'] < len(fluor_names) else f"F{comp[0]['fluor_index']+1}"
            else:
                parts = []
                for c in comp:
                    fname = fluor_names[c['fluor_index']] if c['fluor_index'] < len(fluor_names) else f"F{c['fluor_index']+1}"
                    parts.append(f"{fname}({c['ratio']:.0%})")
                fluor_str = "+".join(parts)
            
            line = f"{obj_id:<6} {obj_type:<14} {pos_str:<14} {intensity:<8.3f} {fluor_str}\n"
            self.object_text.insert(tk.END, line)
    
    def _update_mini_view(self):
        """Update the mini composite view."""
        if not self.selected_object_ids:
            # Hide canvas, show message
            self.mini_canvas_widget.grid_forget()
            self.view_info_label.config(text="Select an object to see zoomed view")
            self.view_info_label.grid(row=0, column=0, pady=20)
            return
        
        data = self.get_data()
        if not data or not data.has_data:
            self.mini_canvas_widget.grid_forget()
            self.view_info_label.config(text="No data available")
            self.view_info_label.grid(row=0, column=0, pady=20)
            return
        
        # Get selected objects
        selected_objs = [obj for obj in self.all_objects 
                        if obj['id'] in self.selected_object_ids]
        
        if not selected_objs:
            return
        
        # Calculate bounding box for selected objects
        positions = [obj['position'] for obj in selected_objs]
        positions = np.array(positions)
        
        if len(self.selected_object_ids) == 1:
            # Single object: tight crop
            center_y, center_x = positions[0]
            crop_size = 20  # 20 pixels on each side
        else:
            # Multiple objects: show all
            center_y = np.mean(positions[:, 0])
            center_x = np.mean(positions[:, 1])
            # Calculate size to fit all objects
            max_dist = np.max(np.abs(positions - [center_y, center_x]))
            crop_size = int(max_dist + 10)  # Add padding
            crop_size = max(15, min(crop_size, 50))  # Clamp to reasonable range
        
        # Generate mini composite
        try:
            mini_rgb = self._generate_mini_composite(data, center_y, center_x, crop_size)
            
            # Update plot
            self.mini_ax.clear()
            self.mini_ax.imshow(mini_rgb, origin='upper')
            self.mini_ax.set_xticks([])
            self.mini_ax.set_yticks([])
            
            # Draw crosshair at center
            h, w = mini_rgb.shape[:2]
            self.mini_ax.plot([w/2], [h/2], '+', color='yellow', markersize=10, markeredgewidth=2)
            
            # Mark all selected objects
            for pos in positions:
                # Convert to mini view coordinates
                rel_y = pos[0] - center_y + crop_size
                rel_x = pos[1] - center_x + crop_size
                if 0 <= rel_y < 2*crop_size and 0 <= rel_x < 2*crop_size:
                    self.mini_ax.plot([rel_x], [rel_y], 'o', 
                                     markerfacecolor='none', markeredgecolor='cyan', 
                                     markersize=8, markeredgewidth=1.5)
            
            self.mini_ax.set_title(f"Zoomed View ({2*crop_size}×{2*crop_size} px)", 
                                  fontsize=9)
            
            self.mini_figure.tight_layout()
            self.mini_canvas.draw()
            
            # Show canvas, hide message
            self.view_info_label.grid_forget()
            self.mini_canvas_widget.grid(row=0, column=0, sticky='nsew', padx=4, pady=4)
            
        except Exception as e:
            self.view_info_label.config(text=f"Error generating view: {str(e)}")
            self.view_info_label.grid(row=0, column=0, pady=20)
            self.mini_canvas_widget.grid_forget()
    
    def _generate_mini_composite(self, data, center_y: float, center_x: float, 
                                 crop_size: int) -> np.ndarray:
        """Generate a cropped composite image centered on position."""
        # Get the full composite from main view (we'll need to access this)
        # For now, regenerate a simple version
        from ..views.composite.composite_view import create_composite_rgb
        
        Y = data.Y
        M = data.M
        H, W = data.field.shape
        
        # Get active channels (we need access to state)
        # For simplicity, use all channels for now
        # TODO: respect active channel selection
        
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
        
        return cropped
    
    def _clear_selection(self):
        """Clear current selection."""
        self.selected_object_ids = []
        self._update_display()

