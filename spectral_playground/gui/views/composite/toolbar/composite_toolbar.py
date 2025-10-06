"""Toolbar creation and management for composite view."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

from ..enums import SpectralMode


class CompositeToolbar:
    """Manages toolbar creation and button states."""
    
    def __init__(self, parent: tk.Widget, figure, canvas, 
                 spectral_mode_callback, visual_settings_callback,
                 object_overlay_callback, area_select_callback,
                 show_objects_var):
        """Initialize composite toolbar.
        
        Args:
            parent: Parent widget
            figure: Matplotlib figure
            canvas: Matplotlib canvas
            spectral_mode_callback: Callback for spectral mode changes
            visual_settings_callback: Callback for visual settings button
            object_overlay_callback: Callback for object overlay toggle
            area_select_callback: Callback for area select toggle
            show_objects_var: BooleanVar for show objects checkbox
        """
        self.figure = figure
        self.canvas = canvas
        self.spectral_mode_callback = spectral_mode_callback
        self.visual_settings_callback = visual_settings_callback
        self.object_overlay_callback = object_overlay_callback
        self.area_select_callback = area_select_callback
        self.show_objects_var = show_objects_var
        
        self.toolbar_frame = tk.Frame(parent)
        self.toolbar_frame.grid(row=2, column=0, sticky=tk.EW, pady=(2, 0))
        
        # Standard matplotlib toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.pack(side=tk.LEFT)
        
        # Disable matplotlib's coordinate display
        try:
            self.toolbar.set_message = lambda s: None
        except:
            pass
        
        # Create custom tools
        self._create_spectral_tools()
        self._create_visual_settings_button()
    
    def _create_spectral_tools(self) -> None:
        """Create spectral analysis tool buttons."""
        separator = ttk.Separator(self.toolbar_frame, orient='vertical')
        separator.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 5))
        
        spectral_frame = tk.Frame(self.toolbar_frame)
        spectral_frame.pack(side=tk.LEFT)
        
        tk.Label(spectral_frame, text="Spectral:", font=('TkDefaultFont', 8)).pack(side=tk.LEFT, padx=(0, 5))
        
        self.pixel_btn = tk.Button(spectral_frame, text="📍", width=3, relief=tk.RAISED,
                                   command=lambda: self.spectral_mode_callback(SpectralMode.PIXEL))
        self.pixel_btn.pack(side=tk.LEFT, padx=1)
        
        self.line_btn = tk.Button(spectral_frame, text="📏", width=3, relief=tk.RAISED,
                                  command=lambda: self.spectral_mode_callback(SpectralMode.LINE))
        self.line_btn.pack(side=tk.LEFT, padx=1)
        
        self.area_btn = tk.Button(spectral_frame, text="⬛", width=3, relief=tk.RAISED,
                                  command=lambda: self.spectral_mode_callback(SpectralMode.AREA))
        self.area_btn.pack(side=tk.LEFT, padx=1)
        
        self.clear_btn = tk.Button(spectral_frame, text="✕", width=3, relief=tk.RAISED,
                                   command=lambda: self.spectral_mode_callback(SpectralMode.NONE))
        self.clear_btn.pack(side=tk.LEFT, padx=(5, 0))
        
        # Object selection tools
        obj_separator = ttk.Separator(self.toolbar_frame, orient='vertical')
        obj_separator.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5))
        
        obj_frame = tk.Frame(self.toolbar_frame)
        obj_frame.pack(side=tk.LEFT)
        
        tk.Label(obj_frame, text="Objects:", font=('TkDefaultFont', 8)).pack(side=tk.LEFT, padx=(0, 5))
        
        self.show_objects_check = ttk.Checkbutton(obj_frame, text="Show", 
                                                  variable=self.show_objects_var,
                                                  command=self.object_overlay_callback)
        self.show_objects_check.pack(side=tk.LEFT, padx=1)
        
        self.area_select_btn = tk.Button(obj_frame, text="📦", width=3, relief=tk.RAISED,
                                         command=self.area_select_callback)
        self.area_select_btn.pack(side=tk.LEFT, padx=(5, 0))
        tk.Label(obj_frame, text="Area Select", font=('TkDefaultFont', 7)).pack(side=tk.LEFT, padx=(2, 0))
    
    def _create_visual_settings_button(self) -> None:
        """Create visual settings button."""
        visual_separator = ttk.Separator(self.toolbar_frame, orient='vertical')
        visual_separator.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5))
        
        self.visual_settings_btn = tk.Button(self.toolbar_frame, text="⚙️", width=3, relief=tk.RAISED,
                                           command=self.visual_settings_callback)
        self.visual_settings_btn.pack(side=tk.LEFT, padx=(0, 10))
    
    def update_spectral_mode_buttons(self, mode: SpectralMode) -> None:
        """Update button appearances for spectral mode.
        
        Args:
            mode: Current spectral mode
        """
        buttons = [self.pixel_btn, self.line_btn, self.area_btn, self.clear_btn]
        for btn in buttons:
            btn.config(relief=tk.RAISED, bg='SystemButtonFace')
        
        if mode == SpectralMode.PIXEL:
            self.pixel_btn.config(relief=tk.SUNKEN, bg='lightblue')
        elif mode == SpectralMode.LINE:
            self.line_btn.config(relief=tk.SUNKEN, bg='lightgreen')
        elif mode == SpectralMode.AREA:
            self.area_btn.config(relief=tk.SUNKEN, bg='lightyellow')
    
    def update_area_select_button(self, active: bool) -> None:
        """Update area select button appearance.
        
        Args:
            active: Whether area select mode is active
        """
        if active:
            self.area_select_btn.config(relief=tk.SUNKEN, bg='lightgreen')
        else:
            self.area_select_btn.config(relief=tk.RAISED, bg='SystemButtonFace')
    
    def override_home_button(self, custom_home_func) -> None:
        """Override toolbar's home button with custom function.
        
        Args:
            custom_home_func: Custom home function to use
        """
        self.toolbar.home = custom_home_func
    
    def deactivate_tools(self) -> None:
        """Deactivate matplotlib toolbar tools."""
        if hasattr(self.toolbar, '_active') and self.toolbar._active:
            self.toolbar._active = None

