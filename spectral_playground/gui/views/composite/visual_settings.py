"""Visual settings management for composite view."""

from __future__ import annotations
from typing import Optional, Callable
import tkinter as tk
from tkinter import ttk


class VisualSettingsManager:
    """Manages visual settings for the composite view."""
    
    def __init__(self, parent_widget: tk.Widget, on_changed: Optional[Callable] = None):
        """Initialize visual settings manager.
        
        Args:
            parent_widget: Parent widget for timer callbacks
            on_changed: Callback when settings change
        """
        self.parent_widget = parent_widget
        self._on_changed = on_changed
        
        # Settings state
        self.normalization_mode = tk.StringVar(value="per_channel")  # "per_channel" or "global"
        self.percentile_threshold = tk.DoubleVar(value=100.0)  # 85-100
        self.gamma_correction = tk.DoubleVar(value=1.0)  # 0.3-2.0
        self.use_log_scaling = tk.BooleanVar(value=False)
        
        # Debounce timer for slider updates (performance optimization)
        self._slider_update_timer = None
        
        # Label references for value updates
        self.percentile_value_label: Optional[tk.Label] = None
        self.gamma_value_label: Optional[tk.Label] = None
    
    def show_settings_dialog(self) -> None:
        """Show visual settings popup window."""
        window = tk.Toplevel()
        window.title('Visual Settings')
        window.geometry('500x650')
        window.transient()
        window.grab_set()
        window.resizable(True, True)
        
        main_frame = tk.Frame(window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Channel Normalization Section
        self._create_normalization_section(main_frame)
        
        # Intensity Mapping Section
        self._create_intensity_mapping_section(main_frame)
        
        # Buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Close", command=window.destroy).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="Reset to Defaults", 
                  command=self.reset_to_defaults).pack(side=tk.RIGHT, padx=(0, 10))
    
    def _create_normalization_section(self, parent: tk.Frame) -> None:
        """Create normalization settings section."""
        norm_frame = ttk.LabelFrame(parent, text="Channel Normalization", padding=10)
        norm_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(norm_frame, text="Controls how channel intensities are normalized for display:", 
                font=('TkDefaultFont', 9)).pack(anchor='w', pady=(0, 10))
        
        ttk.Radiobutton(norm_frame, text="Per-channel (current behavior)", 
                       variable=self.normalization_mode, value="per_channel",
                       command=self._on_settings_changed).pack(anchor='w', pady=2)
        
        tk.Label(norm_frame, text="   • Each channel normalized to its own maximum", 
                font=('TkDefaultFont', 8), fg='gray').pack(anchor='w')
        tk.Label(norm_frame, text="   • Weak channels can appear artificially bright", 
                font=('TkDefaultFont', 8), fg='gray').pack(anchor='w', pady=(0, 8))
        
        ttk.Radiobutton(norm_frame, text="Global normalization", 
                       variable=self.normalization_mode, value="global",
                       command=self._on_settings_changed).pack(anchor='w', pady=2)
        
        tk.Label(norm_frame, text="   • All channels normalized to strongest overall signal", 
                font=('TkDefaultFont', 8), fg='gray').pack(anchor='w')
        tk.Label(norm_frame, text="   • True relative channel strengths preserved", 
                font=('TkDefaultFont', 8), fg='gray').pack(anchor='w')
    
    def _create_intensity_mapping_section(self, parent: tk.Frame) -> None:
        """Create intensity mapping controls section."""
        intensity_frame = ttk.LabelFrame(parent, text="Intensity Mapping", padding=10)
        intensity_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Percentile Threshold Slider
        self._create_percentile_slider(intensity_frame)
        
        ttk.Separator(intensity_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Gamma Correction Slider
        self._create_gamma_slider(intensity_frame)
        
        ttk.Separator(intensity_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Logarithmic Scaling Checkbox
        self._create_log_scaling_checkbox(intensity_frame)
    
    def _create_percentile_slider(self, parent: tk.Frame) -> None:
        """Create percentile threshold slider."""
        percentile_label_frame = tk.Frame(parent)
        percentile_label_frame.pack(fill=tk.X, pady=(0, 5))
        tk.Label(percentile_label_frame, text="Percentile Threshold:", 
                font=('TkDefaultFont', 9, 'bold')).pack(side=tk.LEFT)
        self.percentile_value_label = tk.Label(percentile_label_frame, 
                                              text=f"{self.percentile_threshold.get():.1f}%",
                                              font=('TkDefaultFont', 9))
        self.percentile_value_label.pack(side=tk.RIGHT)
        
        tk.Label(parent, text="Controls saturation point for intensity normalization",
                font=('TkDefaultFont', 8), fg='gray').pack(anchor='w')
        
        percentile_slider_frame = tk.Frame(parent)
        percentile_slider_frame.pack(fill=tk.X, pady=(5, 10))
        tk.Label(percentile_slider_frame, text="85%", font=('TkDefaultFont', 7)).pack(side=tk.LEFT)
        percentile_slider = ttk.Scale(percentile_slider_frame, from_=85.0, to=100.0,
                                     variable=self.percentile_threshold, orient=tk.HORIZONTAL,
                                     command=lambda v: self._on_percentile_changed())
        percentile_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tk.Label(percentile_slider_frame, text="100%", font=('TkDefaultFont', 7)).pack(side=tk.RIGHT)
    
    def _create_gamma_slider(self, parent: tk.Frame) -> None:
        """Create gamma correction slider."""
        gamma_label_frame = tk.Frame(parent)
        gamma_label_frame.pack(fill=tk.X, pady=(0, 5))
        tk.Label(gamma_label_frame, text="Gamma Correction:", 
                font=('TkDefaultFont', 9, 'bold')).pack(side=tk.LEFT)
        self.gamma_value_label = tk.Label(gamma_label_frame, 
                                          text=f"{self.gamma_correction.get():.2f}",
                                          font=('TkDefaultFont', 9))
        self.gamma_value_label.pack(side=tk.RIGHT)
        
        tk.Label(parent, text="Adjusts brightness curve (< 1.0 = brighter, > 1.0 = darker)",
                font=('TkDefaultFont', 8), fg='gray').pack(anchor='w')
        
        gamma_slider_frame = tk.Frame(parent)
        gamma_slider_frame.pack(fill=tk.X, pady=(5, 10))
        tk.Label(gamma_slider_frame, text="0.3", font=('TkDefaultFont', 7)).pack(side=tk.LEFT)
        gamma_slider = ttk.Scale(gamma_slider_frame, from_=0.3, to=2.0,
                                variable=self.gamma_correction, orient=tk.HORIZONTAL,
                                command=lambda v: self._on_gamma_changed())
        gamma_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tk.Label(gamma_slider_frame, text="2.0", font=('TkDefaultFont', 7)).pack(side=tk.RIGHT)
    
    def _create_log_scaling_checkbox(self, parent: tk.Frame) -> None:
        """Create logarithmic scaling checkbox."""
        log_frame = tk.Frame(parent)
        log_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Checkbutton(log_frame, text="Use Logarithmic Scaling", 
                       variable=self.use_log_scaling,
                       command=self._on_settings_changed).pack(anchor='w')
        
        tk.Label(parent, text="Compresses wide dynamic range (use for very bright spots)",
                font=('TkDefaultFont', 8), fg='gray').pack(anchor='w', padx=(20, 0))
    
    def _on_settings_changed(self) -> None:
        """Called when visual settings change - trigger composite redraw."""
        if self._on_changed:
            self._on_changed()
    
    def _on_percentile_changed(self) -> None:
        """Update percentile label immediately, debounce redraw."""
        if self.percentile_value_label:
            self.percentile_value_label.config(text=f"{self.percentile_threshold.get():.1f}%")
        # Debounce: cancel pending update and schedule new one
        if self._slider_update_timer:
            self.parent_widget.after_cancel(self._slider_update_timer)
        self._slider_update_timer = self.parent_widget.after(200, self._on_settings_changed)
    
    def _on_gamma_changed(self) -> None:
        """Update gamma label immediately, debounce redraw."""
        if self.gamma_value_label:
            self.gamma_value_label.config(text=f"{self.gamma_correction.get():.2f}")
        # Debounce: cancel pending update and schedule new one
        if self._slider_update_timer:
            self.parent_widget.after_cancel(self._slider_update_timer)
        self._slider_update_timer = self.parent_widget.after(200, self._on_settings_changed)
    
    def reset_to_defaults(self) -> None:
        """Reset visual settings to defaults."""
        self.normalization_mode.set("per_channel")
        self.percentile_threshold.set(100.0)
        self.gamma_correction.set(1.0)
        self.use_log_scaling.set(False)
        if self.percentile_value_label:
            self.percentile_value_label.config(text="100.0%")
        if self.gamma_value_label:
            self.gamma_value_label.config(text="1.00")
        self._on_settings_changed()
