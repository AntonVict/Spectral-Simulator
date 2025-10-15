"""Controls panel for statistics analysis parameters."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Any, Callable, Optional

from .parameter_widgets import create_scrollable_frame
from ..utils import auto_fit_parameters_from_scene


class ControlsPanel(ttk.Frame):
    """Panel containing all analysis parameter controls."""
    
    def __init__(
        self,
        parent: tk.Widget,
        state,
        log_callback: Callable[[str], None],
        run_analysis_callback: Optional[Callable[[], None]] = None,
        export_callback: Optional[Callable[[], None]] = None
    ):
        """Initialize controls panel.
        
        Args:
            parent: Parent widget
            state: Playground state
            log_callback: Function to log messages
            run_analysis_callback: Callback for Run Analysis button
            export_callback: Callback for Export button
        """
        super().__init__(parent)
        
        self.state = state
        self.log = log_callback
        self._run_analysis_callback = run_analysis_callback
        self._export_callback = export_callback
        
        # Initialize all parameter variables
        self.active_policy = tk.StringVar(value='overlap')
        self.overlap_mode = tk.StringVar(value='continuous')
        self.intensity = tk.DoubleVar(value=0.001)
        self.neighbor_threshold = tk.IntVar(value=1)
        self.box_size = tk.IntVar(value=8)
        self.box_threshold = tk.IntVar(value=2)
        self.count_mode = tk.StringVar(value='germ')
        self.radius_mean = tk.DoubleVar(value=3.0)
        self.radius_std = tk.DoubleVar(value=0.6)
        self.radius_min = tk.DoubleVar(value=2.0)
        self.radius_max = tk.DoubleVar(value=5.0)
        
        # Build UI
        self._build_ui()
        
        # Set up policy change callback
        self.active_policy.trace_add('write', lambda *args: self.update_policy_visibility())
    
    def _build_ui(self) -> None:
        """Build the controls panel UI."""
        # Scrollable container
        canvas, scrollable_frame = create_scrollable_frame(self)
        
        # Auto-fit button at top
        autofit_frame = ttk.Frame(scrollable_frame)
        autofit_frame.pack(fill=tk.X, padx=4, pady=(4, 8))
        
        ttk.Button(
            autofit_frame,
            text='⚙ Auto-Fit All from Scene',
            command=self._auto_fit_from_scene
        ).pack(fill=tk.X)
        
        # Status label for fitted values
        self.fitted_status_label = ttk.Label(
            autofit_frame,
            text='',
            font=('TkDefaultFont', 8),
            foreground='#2a7f2a'
        )
        self.fitted_status_label.pack(anchor='w', pady=(2, 0))
        
        # Policy selection
        policy_frame = ttk.LabelFrame(scrollable_frame, text='Select Overlap Policy')
        policy_frame.pack(fill=tk.X, padx=4, pady=4)
        
        ttk.Radiobutton(
            policy_frame,
            text='Neighbor Count Policy',
            variable=self.active_policy,
            value='overlap'
        ).pack(anchor='w', padx=4)
        
        ttk.Radiobutton(
            policy_frame,
            text='Region Density Policy',
            variable=self.active_policy,
            value='box'
        ).pack(anchor='w', padx=4)
        
        # Overlap Detection Mode
        overlap_mode_frame = ttk.LabelFrame(scrollable_frame, text='Overlap Detection Mode')
        overlap_mode_frame.pack(fill=tk.X, padx=4, pady=4)
        
        ttk.Radiobutton(
            overlap_mode_frame,
            text='Continuous (sub-pixel)',
            variable=self.overlap_mode,
            value='continuous'
        ).pack(anchor='w', padx=4)
        
        ttk.Radiobutton(
            overlap_mode_frame,
            text='Discrete (pixelated)',
            variable=self.overlap_mode,
            value='pixelated'
        ).pack(anchor='w', padx=4)
        
        ttk.Label(
            overlap_mode_frame,
            text='Continuous: Exact geometric overlap\nPixel-Based: Simulates image analysis',
            font=('TkDefaultFont', 8, 'italic'),
            foreground='#555',
            justify=tk.LEFT
        ).pack(anchor='w', padx=4, pady=(2, 4))
        
        # Spatial intensity (λ)
        intensity_frame = ttk.LabelFrame(scrollable_frame, text='Spatial Density')
        intensity_frame.pack(fill=tk.X, padx=4, pady=4)
        
        ttk.Label(intensity_frame, text='λ (objects per pixel²):').pack(anchor='w', padx=4)
        ttk.Entry(intensity_frame, textvariable=self.intensity, width=12).pack(
            anchor='w', padx=4, pady=(0, 4)
        )
        
        # Neighbor Count Policy parameters
        self.neighbor_frame = ttk.LabelFrame(scrollable_frame, text='Neighbor Count Policy Parameters')
        self.neighbor_frame.pack(fill=tk.X, padx=4, pady=4)
        
        ttk.Label(self.neighbor_frame, text='Neighbor threshold (m):').pack(anchor='w', padx=4)
        ttk.Spinbox(
            self.neighbor_frame,
            from_=1,
            to=20,
            textvariable=self.neighbor_threshold,
            width=10
        ).pack(anchor='w', padx=4)
        
        ttk.Label(
            self.neighbor_frame,
            text='Discard if ≥ m neighbors overlap',
            font=('TkDefaultFont', 8, 'italic'),
            foreground='#555'
        ).pack(anchor='w', padx=4, pady=(2, 4))
        
        # Region Density Policy parameters
        self.box_frame = ttk.LabelFrame(scrollable_frame, text='Region Density Policy Parameters')
        self.box_frame.pack(fill=tk.X, padx=4, pady=4)
        
        ttk.Label(self.box_frame, text='Box side length (a, pixels):').pack(anchor='w', padx=4)
        
        scale = ttk.Scale(
            self.box_frame,
            from_=2,
            to=32,
            variable=self.box_size,
            orient=tk.HORIZONTAL,
            command=lambda _: self._update_box_area_display()
        )
        scale.pack(fill=tk.X, padx=4)
        
        # Show current value and area
        box_value_frame = ttk.Frame(self.box_frame)
        box_value_frame.pack(fill=tk.X, padx=4)
        ttk.Label(box_value_frame, text='a = ').pack(side=tk.LEFT)
        ttk.Label(box_value_frame, textvariable=self.box_size).pack(side=tk.LEFT)
        ttk.Label(box_value_frame, text=' → Box area: ').pack(side=tk.LEFT)
        self.box_area_label = ttk.Label(box_value_frame, text='64 px²', foreground='#2a7f2a')
        self.box_area_label.pack(side=tk.LEFT)
        
        ttk.Label(self.box_frame, text='Occupancy threshold (k₀):').pack(anchor='w', padx=4, pady=(6,0))
        ttk.Spinbox(
            self.box_frame,
            from_=1,
            to=20,
            textvariable=self.box_threshold,
            width=10
        ).pack(anchor='w', padx=4)
        
        ttk.Label(
            self.box_frame,
            text='Crowded if ≥ k₀ objects in box',
            font=('TkDefaultFont', 8, 'italic'),
            foreground='#555'
        ).pack(anchor='w', padx=4, pady=(2, 6))
        
        ttk.Label(self.box_frame, text='Counting mode:').pack(anchor='w', padx=4, pady=(6,0))
        ttk.Radiobutton(
            self.box_frame,
            text='Germ (centers only)',
            variable=self.count_mode,
            value='germ'
        ).pack(anchor='w', padx=4)
        ttk.Radiobutton(
            self.box_frame,
            text='Intersection (disc overlap)',
            variable=self.count_mode,
            value='intersection'
        ).pack(anchor='w', padx=4, pady=(0, 4))
        
        # Radius distribution parameters
        dist_frame = ttk.LabelFrame(scrollable_frame, text='Radius Distribution')
        dist_frame.pack(fill=tk.X, padx=4, pady=4)
        
        ttk.Label(dist_frame, text='Mean (μ, pixels):').pack(anchor='w', padx=4)
        ttk.Entry(dist_frame, textvariable=self.radius_mean, width=10).pack(
            anchor='w', padx=4
        )
        
        ttk.Label(dist_frame, text='Std Dev (σ, pixels):').pack(anchor='w', padx=4, pady=(4, 0))
        ttk.Entry(dist_frame, textvariable=self.radius_std, width=10).pack(
            anchor='w', padx=4
        )
        
        ttk.Label(dist_frame, text='Min radius (pixels):').pack(anchor='w', padx=4, pady=(4, 0))
        ttk.Entry(dist_frame, textvariable=self.radius_min, width=10).pack(
            anchor='w', padx=4
        )
        
        ttk.Label(dist_frame, text='Max radius (pixels):').pack(anchor='w', padx=4, pady=(4, 0))
        ttk.Entry(dist_frame, textvariable=self.radius_max, width=10).pack(
            anchor='w', padx=4, pady=(0, 4)
        )
        
        # Action buttons
        btn_frame = ttk.Frame(scrollable_frame)
        btn_frame.pack(fill=tk.X, padx=4, pady=8)
        
        ttk.Button(
            btn_frame,
            text='Run Analysis',
            command=self._on_run_analysis
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            btn_frame,
            text='Export Results (CSV)',
            command=self._on_export
        ).pack(fill=tk.X, pady=2)
    
    def update_policy_visibility(self) -> None:
        """Show/hide policy-specific controls based on selection."""
        policy = self.active_policy.get()
        
        if policy == 'overlap':
            self.neighbor_frame.pack(fill=tk.X, padx=4, pady=4)
            self.box_frame.pack_forget()
        else:  # box
            self.box_frame.pack(fill=tk.X, padx=4, pady=4)
            self.neighbor_frame.pack_forget()
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get all current parameter values.
        
        Returns:
            Dictionary of parameter names to values
        """
        return {
            'active_policy': self.active_policy.get(),
            'overlap_mode': self.overlap_mode.get(),
            'intensity': self.intensity.get(),
            'neighbor_threshold': self.neighbor_threshold.get(),
            'box_size': self.box_size.get(),
            'box_threshold': self.box_threshold.get(),
            'count_mode': self.count_mode.get(),
            'radius_mean': self.radius_mean.get(),
            'radius_std': self.radius_std.get(),
            'radius_min': self.radius_min.get(),
            'radius_max': self.radius_max.get()
        }
    
    def set_fitted_status(self, text: str) -> None:
        """Update the auto-fit status label.
        
        Args:
            text: Status text to display
        """
        self.fitted_status_label.config(text=text)
    
    def _update_box_area_display(self) -> None:
        """Update the box area label when slider moves."""
        a = self.box_size.get()
        area = a * a
        self.box_area_label.config(text=f'{area} px²')
    
    def _auto_fit_from_scene(self) -> None:
        """Auto-fit parameters from current scene."""
        if not self.state.data.has_geometric_data:
            messagebox.showwarning(
                'No Data',
                'Please generate objects first to auto-fit parameters.'
            )
            return
        
        scene = self.state.data.geometric_scene
        
        try:
            # Get fitted parameters
            params = auto_fit_parameters_from_scene(scene, self.log)
            
            # Update UI
            self.radius_mean.set(params['radius_mean'])
            self.radius_std.set(params['radius_std'])
            self.radius_min.set(params['radius_min'])
            self.radius_max.set(params['radius_max'])
            self.intensity.set(params['lambda_val'])
            
            # Update status
            self.set_fitted_status(params['status_message'])
            
            # Show info
            messagebox.showinfo('Auto-Fit Complete', params['info_message'])
            
        except ValueError as e:
            messagebox.showwarning('Auto-Fit Error', str(e))
    
    def _on_run_analysis(self) -> None:
        """Handle Run Analysis button click."""
        if self._run_analysis_callback:
            self._run_analysis_callback()
    
    def _on_export(self) -> None:
        """Handle Export button click."""
        if self._export_callback:
            self._export_callback()

