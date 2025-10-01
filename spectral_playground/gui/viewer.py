from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk
from typing import Callable, Iterable, List

from .state import PlaygroundState
from .views.composite import CompositeView
from .views.inspector import ObjectInspectorView


class ViewerPanel(ttk.Frame):
    """Right-hand panel containing the composite view and controls."""

    def __init__(
        self,
        parent: tk.Widget,
        state: PlaygroundState,
        on_generate: Callable[[], None],
        on_load: Callable[[], None],
        on_save_dataset: Callable[[], None],
        on_save_composite: Callable[[], None],
        on_export_plots: Callable[[], None],
        on_open_folder: Callable[[], None],
        on_channels_changed: Callable[[], None],
        on_expand_composite: Callable[[], None],
    ) -> None:
        super().__init__(parent)
        self.state = state
        self.on_channels_changed = on_channels_changed

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        main_frame = ttk.Frame(self)
        main_frame.grid(row=0, column=0, sticky='nsew', padx=8, pady=8)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=0)
        main_frame.rowconfigure(0, weight=1)

        image_frame = ttk.Frame(main_frame)
        image_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 8))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(1, weight=1)
        # Ensure row 2 (toolbar) doesn't take extra space
        image_frame.rowconfigure(2, weight=0)

        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=0, column=1, sticky='ns')

        actions_group = ttk.LabelFrame(controls_frame, text='Actions')
        actions_group.pack(fill=tk.X, padx=4, pady=4)

        ttk.Button(actions_group, text='Generate Data', command=on_generate, width=14).pack(pady=2)
        ttk.Button(actions_group, text='Load Dataset', command=on_load, width=14).pack(pady=2)
        ttk.Button(actions_group, text='Save Dataset', command=on_save_dataset, width=14).pack(pady=2)
        ttk.Button(actions_group, text='Save Composite', command=on_save_composite, width=14).pack(pady=2)
        ttk.Button(actions_group, text='Export Plots', command=on_export_plots, width=14).pack(pady=2)
        ttk.Button(actions_group, text='Open Save Folder', command=on_open_folder, width=14).pack(pady=(2, 0))

        channel_group = ttk.LabelFrame(controls_frame, text='Channels')
        channel_group.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.channel_checks_frame = ttk.Frame(channel_group)
        self.channel_checks_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.channel_vars: List[tk.BooleanVar] = []

        ttk.Button(channel_group, text='Select All', command=self.select_all_channels).pack(fill=tk.X, padx=4, pady=(4, 2))
        ttk.Button(channel_group, text='Select None', command=self.select_no_channels).pack(fill=tk.X, padx=4, pady=(0, 4))

        top_controls = ttk.Frame(image_frame)
        top_controls.grid(row=0, column=0, sticky='ew')
        ttk.Label(top_controls, text='Composite View').pack(side=tk.LEFT)
        ttk.Button(top_controls, text='Inspector', command=self._open_inspector, width=12).pack(side=tk.RIGHT, padx=2)
        ttk.Button(top_controls, text='Expand', command=on_expand_composite, width=12).pack(side=tk.RIGHT)

        self.composite_view = CompositeView(image_frame, on_visual_settings_changed=self._on_composite_visual_settings_changed)
        self.composite_view.set_object_selection_callback(self._on_object_selection_changed)
        self.inspector_window = None

    def set_channels(self, names: Iterable[str]) -> None:
        for widget in self.channel_checks_frame.winfo_children():
            widget.destroy()
        self.channel_vars.clear()
        for idx, name in enumerate(names):
            var = tk.BooleanVar(value=True)
            chk = ttk.Checkbutton(
                self.channel_checks_frame,
                text=name,
                variable=var,
                command=self._channels_changed,
            )
            chk.pack(anchor='w', pady=1)
            self.channel_vars.append(var)
        self._channels_changed()

    def _channels_changed(self) -> None:
        flags = tuple(var.get() for var in self.channel_vars)
        self.state.selections.update_channels(flags)
        self.on_channels_changed()

    def select_all_channels(self) -> None:
        for var in self.channel_vars:
            var.set(True)
        self._channels_changed()

    def select_no_channels(self) -> None:
        for var in self.channel_vars:
            var.set(False)
        self._channels_changed()

    def active_channel_flags(self) -> List[bool]:
        return [var.get() for var in self.channel_vars]
    
    def _on_composite_visual_settings_changed(self) -> None:
        """Called when composite visual settings change - trigger redraw."""
        # Clear RGB cache to force re-render with new visual settings
        self.composite_view._rgb_cache_dict.clear()
        self.on_channels_changed()  # This will trigger update_visualisation in main_gui
    
    def _open_inspector(self, object_ids: list = None) -> None:
        """Open the Object Inspector window.
        
        Args:
            object_ids: Optional list of object IDs to select when opening
        """
        if self.inspector_window is not None and self.inspector_window.winfo_exists():
            # Window already open, just raise it and update selection
            self.inspector_window.lift()
            self.inspector_window.focus()
            
            # If object_ids provided, select them in existing inspector
            if object_ids is not None:
                for child in self.inspector_window.winfo_children():
                    if isinstance(child, ObjectInspectorView):
                        child.set_selected_objects(object_ids)
                        break
            return
        
        # Create new inspector window
        self.inspector_window = tk.Toplevel(self)
        self.inspector_window.title("Object Inspector")
        self.inspector_window.geometry("1000x700")
        
        # Get callbacks for the inspector
        def get_data():
            return self.state.data
        
        def get_fluorophore_names():
            if self.state.data and self.state.data.spectral:
                return [f.name for f in self.state.data.spectral.fluors]
            return []
        
        # Create inspector view
        inspector = ObjectInspectorView(
            self.inspector_window,
            get_data_callback=get_data,
            get_fluorophore_names_callback=get_fluorophore_names
        )
        inspector.pack(fill=tk.BOTH, expand=True)
        
        # Auto-refresh on open
        inspector.refresh_objects()
        
        # Select objects if IDs provided
        if object_ids is not None:
            inspector.set_selected_objects(object_ids)
    
    def refresh_inspector(self) -> None:
        """Refresh the inspector window if it's open."""
        if self.inspector_window is not None and self.inspector_window.winfo_exists():
            # Find the inspector view widget
            for child in self.inspector_window.winfo_children():
                if isinstance(child, ObjectInspectorView):
                    child.refresh_objects()
                    break
    
    def _on_object_selection_changed(self, object_ids: list) -> None:
        """Handle object selection changes from CompositeView."""
        # Propagate to parent (main GUI) which will update QuickInspector
        parent = self.master
        if hasattr(parent, '_on_composite_object_selection'):
            parent._on_composite_object_selection(object_ids)
