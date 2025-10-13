"""Statistical Analysis View for crowding metrics and theoretical predictions."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from spectral_playground.gui.state import PlaygroundState
from spectral_playground.core.statistics import (
    RadiusDistribution,
    BooleanModelTheory,
    CrowdingAnalyzer
)


class StatisticsView(ttk.Frame):
    """Statistical analysis dashboard with numbers-first reporting.
    
    Provides controls for configuring analysis parameters, displays
    key metrics in a scoreboard, and shows theoretical vs empirical
    comparisons in plots.
    """
    
    def __init__(
        self,
        parent: tk.Widget,
        state: PlaygroundState,
        log_callback: Callable[[str], None],
        open_inspector_callback: Optional[Callable[[list], None]] = None
    ):
        """Initialize statistics view.
        
        Args:
            parent: Parent widget
            state: Playground state
            log_callback: Function to log messages
            open_inspector_callback: Function to open Inspector with object IDs
        """
        super().__init__(parent)
        self.state = state
        self.log = log_callback
        self.open_inspector = open_inspector_callback
        
        # Analysis results cache
        self.last_results: Optional[Dict[str, Any]] = None
        
        # Configure grid layout: Controls | Scoreboard | Plots
        self.columnconfigure(0, weight=1, minsize=250)
        self.columnconfigure(1, weight=1, minsize=300)
        self.columnconfigure(2, weight=2, minsize=400)
        self.rowconfigure(0, weight=1)
        
        self._build_controls()
        self._build_scoreboard()
        self._build_plots()
    
    def _build_controls(self) -> None:
        """Build control panel with analysis parameters."""
        controls_frame = ttk.LabelFrame(self, text='Analysis Parameters')
        controls_frame.grid(row=0, column=0, sticky='nsew', padx=4, pady=4)
        
        # Scrollable container
        canvas = tk.Canvas(controls_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(controls_frame, orient='vertical', command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            '<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all'))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Auto-fit button at top (prominent placement)
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
        policy_frame = ttk.LabelFrame(scrollable_frame, text='Select Discard Policy')
        policy_frame.pack(fill=tk.X, padx=4, pady=4)
        
        ttk.Label(
            policy_frame,
            text='Choose ONE policy to apply:',
            font=('TkDefaultFont', 8, 'bold')
        ).pack(anchor='w', padx=4, pady=(2, 4))
        
        self.active_policy = tk.StringVar(value='overlap')
        
        ttk.Radiobutton(
            policy_frame,
            text='Overlap Policy (geometric neighbor threshold)',
            variable=self.active_policy,
            value='overlap'
        ).pack(anchor='w', padx=4)
        
        ttk.Radiobutton(
            policy_frame,
            text='Box Crowding Policy (region density threshold)',
            variable=self.active_policy,
            value='box'
        ).pack(anchor='w', padx=4)
        
        # Spatial intensity (λ) - now at top level
        intensity_frame = ttk.LabelFrame(scrollable_frame, text='Spatial Intensity')
        intensity_frame.pack(fill=tk.X, padx=4, pady=4)
        
        ttk.Label(intensity_frame, text='λ (objects per pixel²):').pack(anchor='w')
        self.intensity = tk.DoubleVar(value=0.001)
        ttk.Entry(intensity_frame, textvariable=self.intensity, width=12).pack(
            anchor='w', padx=4
        )
        ttk.Label(
            intensity_frame,
            text='(Used for theoretical predictions)',
            font=('TkDefaultFont', 7, 'italic'),
            foreground='gray'
        ).pack(anchor='w', padx=4, pady=(0, 4))
        
        # Box crowding parameters
        box_frame = ttk.LabelFrame(scrollable_frame, text='Box Crowding Policy')
        box_frame.pack(fill=tk.X, padx=4, pady=4)
        
        ttk.Label(
            box_frame,
            text='Divides image into grid; discards objects in crowded boxes',
            font=('TkDefaultFont', 8, 'italic'),
            foreground='#555'
        ).pack(anchor='w', padx=4, pady=(2, 6))
        
        ttk.Label(box_frame, text='Box side length (a, pixels):').pack(anchor='w')
        self.box_size = tk.IntVar(value=8)
        
        # Integer-only scale with command to update area display
        scale = ttk.Scale(
            box_frame,
            from_=2,
            to=32,
            variable=self.box_size,
            orient=tk.HORIZONTAL,
            command=lambda _: self._update_box_area_display()
        )
        scale.pack(fill=tk.X, padx=4)
        
        # Show current value and area
        box_value_frame = ttk.Frame(box_frame)
        box_value_frame.pack(fill=tk.X, padx=4)
        ttk.Label(box_value_frame, text='a = ').pack(side=tk.LEFT)
        ttk.Label(box_value_frame, textvariable=self.box_size).pack(side=tk.LEFT)
        ttk.Label(box_value_frame, text=' → Box area: ').pack(side=tk.LEFT)
        self.box_area_label = ttk.Label(box_value_frame, text='64 px²', foreground='#2a7f2a')
        self.box_area_label.pack(side=tk.LEFT)
        
        ttk.Label(box_frame, text='Occupancy threshold (k₀):').pack(anchor='w', pady=(6,0))
        ttk.Label(
            box_frame,
            text='(Crowded if ≥ k₀ objects in box)',
            font=('TkDefaultFont', 7, 'italic'),
            foreground='gray'
        ).pack(anchor='w', padx=4)
        self.box_threshold = tk.IntVar(value=2)
        ttk.Spinbox(
            box_frame,
            from_=1,
            to=20,
            textvariable=self.box_threshold,
            width=10
        ).pack(anchor='w', padx=4)
        
        ttk.Label(box_frame, text='Counting mode:').pack(anchor='w', pady=(6,0))
        self.count_mode = tk.StringVar(value='germ')
        ttk.Radiobutton(
            box_frame,
            text='Germ (centers only)',
            variable=self.count_mode,
            value='germ'
        ).pack(anchor='w', padx=4)
        ttk.Radiobutton(
            box_frame,
            text='Intersection (disc overlap)',
            variable=self.count_mode,
            value='intersection'
        ).pack(anchor='w', padx=4)
        
        # Theory formula
        ttk.Label(
            box_frame,
            text='Theory: μ = λa² (germ) or λ(a²+4aE[R]+πE[R²]) (intersect)',
            font=('TkDefaultFont', 7),
            foreground='#0066cc'
        ).pack(anchor='w', padx=4, pady=(4, 4))
        
        # Object overlap parameters - RENAMED
        obj_frame = ttk.LabelFrame(scrollable_frame, text='Geometric Overlap Threshold')
        obj_frame.pack(fill=tk.X, padx=4, pady=4)
        
        ttk.Label(
            obj_frame,
            text='Discards objects with too many overlapping neighbors',
            font=('TkDefaultFont', 8, 'italic'),
            foreground='#555'
        ).pack(anchor='w', padx=4, pady=(2, 6))
        
        ttk.Label(obj_frame, text='Neighbor threshold (m):').pack(anchor='w')
        ttk.Label(
            obj_frame,
            text='(Discard if ≥ m neighbors overlap)',
            font=('TkDefaultFont', 7, 'italic'),
            foreground='gray'
        ).pack(anchor='w', padx=4)
        self.neighbor_threshold = tk.IntVar(value=2)
        ttk.Spinbox(
            obj_frame,
            from_=1,
            to=20,
            textvariable=self.neighbor_threshold,
            width=10
        ).pack(anchor='w', padx=4)
        
        # Theory formula for Palm distribution
        ttk.Label(
            obj_frame,
            text='Theory (Palm): ν = λπ(R² + 2RE[R] + E[R²])',
            font=('TkDefaultFont', 7),
            foreground='#0066cc'
        ).pack(anchor='w', padx=4, pady=(4, 4))
        
        # Radius distribution parameters
        dist_frame = ttk.LabelFrame(scrollable_frame, text='Radius Distribution (for Theory)')
        dist_frame.pack(fill=tk.X, padx=4, pady=4)
        
        ttk.Label(
            dist_frame,
            text='(Theoretical predictions only. Empirical uses actual radii.)',
            font=('TkDefaultFont', 7, 'italic'),
            foreground='gray'
        ).pack(anchor='w', padx=4, pady=(0,4))
        
        ttk.Label(dist_frame, text='Mean (μ, pixels):').pack(anchor='w')
        self.radius_mean = tk.DoubleVar(value=3.0)
        ttk.Entry(dist_frame, textvariable=self.radius_mean, width=10).pack(
            anchor='w', padx=4
        )
        
        ttk.Label(dist_frame, text='Std Dev (σ, pixels):').pack(anchor='w')
        self.radius_std = tk.DoubleVar(value=0.6)
        ttk.Entry(dist_frame, textvariable=self.radius_std, width=10).pack(
            anchor='w', padx=4
        )
        
        ttk.Label(dist_frame, text='Min radius (pixels):').pack(anchor='w')
        self.radius_min = tk.DoubleVar(value=2.0)
        ttk.Entry(dist_frame, textvariable=self.radius_min, width=10).pack(
            anchor='w', padx=4
        )
        
        ttk.Label(dist_frame, text='Max radius (pixels):').pack(anchor='w')
        self.radius_max = tk.DoubleVar(value=5.0)
        ttk.Entry(dist_frame, textvariable=self.radius_max, width=10).pack(
            anchor='w', padx=4
        )
        
        # Coverage theory formula
        ttk.Label(
            dist_frame,
            text='Coverage theory: p = 1 - exp(-λπE[R²])',
            font=('TkDefaultFont', 7),
            foreground='#0066cc'
        ).pack(anchor='w', padx=4, pady=(6, 4))
        
        # Action buttons
        btn_frame = ttk.Frame(scrollable_frame)
        btn_frame.pack(fill=tk.X, padx=4, pady=8)
        
        ttk.Button(
            btn_frame,
            text='Run Analysis',
            command=self._run_analysis
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            btn_frame,
            text='Export Results (CSV)',
            command=self._export_csv
        ).pack(fill=tk.X, pady=2)
    
    def _build_scoreboard(self) -> None:
        """Build numbers-first scoreboard panel."""
        scoreboard_frame = ttk.LabelFrame(self, text='Crowding Metrics')
        scoreboard_frame.grid(row=0, column=1, sticky='nsew', padx=4, pady=4)
        
        # Create scrollable container
        canvas = tk.Canvas(scoreboard_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(scoreboard_frame, orient='vertical', command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            '<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all'))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Metrics
        metrics = [
            ('Total Objects', 'total_objects'),
            ('', ''),  # Separator
            ('Crowded Boxes', 'crowded_boxes'),
            ('Discarded (Box Policy)', 'discarded_by_box'),
            ('Discarded (Object Policy)', 'discarded_by_object'),
            ('Discarded (Either Policy)', 'discarded_either'),
            ('', ''),  # Separator
            ('Good Targets (Strict)', 'good_targets_strict'),
            ('Good/Area (per px²)', 'good_per_area_strict'),
            ('', ''),  # Separator
            ('Coverage Fraction', 'coverage_fraction'),
        ]
        
        self.metric_labels = {}
        
        for i, (name, key) in enumerate(metrics):
            if not name:
                # Separator
                ttk.Separator(scrollable_frame, orient='horizontal').grid(
                    row=i, column=0, columnspan=2, sticky='ew', pady=4
                )
                continue
            
            ttk.Label(
                scrollable_frame,
                text=name + ':',
                font=('TkDefaultFont', 10)
            ).grid(row=i, column=0, sticky='w', padx=4, pady=2)
            
            val_label = ttk.Label(
                scrollable_frame,
                text='—',
                font=('TkDefaultFont', 12, 'bold')
            )
            val_label.grid(row=i, column=1, sticky='e', padx=4, pady=2)
            
            if key:
                self.metric_labels[key] = val_label
        
        # Theory comparison section
        theory_frame = ttk.LabelFrame(scrollable_frame, text='Theory vs Empirical')
        theory_frame.grid(row=len(metrics)+1, column=0, columnspan=2, sticky='ew', padx=4, pady=8)
        
        theory_metrics = [
            ('Coverage (Theory)', 'coverage_theory'),
            ('Coverage (Empirical)', 'coverage_empirical'),
            ('', ''),
            ('Expected Good (Object Policy)', 'expected_good_object'),
            ('Actual Good (Object Policy)', 'actual_good_object'),
        ]
        
        for i, (name, key) in enumerate(theory_metrics):
            if not name:
                ttk.Separator(theory_frame, orient='horizontal').grid(
                    row=i, column=0, columnspan=2, sticky='ew', pady=2
                )
                continue
            
            ttk.Label(
                theory_frame,
                text=name + ':',
                font=('TkDefaultFont', 9)
            ).grid(row=i, column=0, sticky='w', padx=4, pady=1)
            
            val_label = ttk.Label(
                theory_frame,
                text='—',
                font=('TkDefaultFont', 10, 'bold')
            )
            val_label.grid(row=i, column=1, sticky='e', padx=4, pady=1)
            
            if key:
                self.metric_labels[key] = val_label
        
        # Inspector buttons
        inspector_frame = ttk.LabelFrame(scrollable_frame, text='Inspect Objects')
        inspector_frame.grid(row=len(metrics)+2, column=0, columnspan=2, sticky='ew', padx=4, pady=8)
        
        ttk.Button(
            inspector_frame,
            text='View Overlapping Objects',
            command=self._view_discarded
        ).pack(fill=tk.X, padx=4, pady=2)
        
        ttk.Button(
            inspector_frame,
            text='View Isolated Objects',
            command=self._view_good
        ).pack(fill=tk.X, padx=4, pady=2)
    
    def _build_plots(self) -> None:
        """Build plots panel."""
        plots_frame = ttk.LabelFrame(self, text='Analysis Plots')
        plots_frame.grid(row=0, column=2, sticky='nsew', padx=4, pady=4)
        
        # Create matplotlib figure with 2 subplots
        self.figure = Figure(figsize=(6, 8), dpi=100)
        self.figure.subplots_adjust(hspace=0.3, left=0.12, right=0.95, top=0.95, bottom=0.08)
        
        self.ax_coverage = self.figure.add_subplot(211)
        self.ax_coverage.set_title('Coverage Analysis')
        self.ax_coverage.set_xlabel('Intensity (λ)')
        self.ax_coverage.set_ylabel('Coverage Fraction')
        self.ax_coverage.grid(True, alpha=0.3)
        
        self.ax_good_targets = self.figure.add_subplot(212)
        self.ax_good_targets.set_title('Good Targets vs Density')
        self.ax_good_targets.set_xlabel('Total Objects (n)')
        self.ax_good_targets.set_ylabel('Good Targets')
        self.ax_good_targets.grid(True, alpha=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=plots_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
    
    def _run_analysis(self) -> None:
        """Run statistical analysis on current data."""
        if not self.state.data.has_geometric_data:
            messagebox.showwarning(
                'No Data',
                'Please generate objects first using the Visualization tab.'
            )
            return
        
        try:
            self.log('Running statistical analysis...')
            
            scene = self.state.data.geometric_scene
            n_objects = len(scene)
            H, W = scene.field_shape
            
            # Get parameters
            a = self.box_size.get()
            k0 = self.box_threshold.get()
            m = self.neighbor_threshold.get()
            mode = self.count_mode.get()
            
            # Performance warnings for large datasets
            grid_H = H // a
            grid_W = W // a
            n_boxes = grid_H * grid_W
            
            # Warn if box grid is very large
            if n_boxes > 100000:
                response = messagebox.askyesno(
                    'Large Grid Warning',
                    f'Analysis will create {n_boxes:,} boxes ({grid_H}×{grid_W}).\n'
                    f'This may take 10-30 seconds.\n\n'
                    f'Tip: Increase box size (a) to {a*2} or {a*4} for faster analysis.\n\n'
                    f'Continue anyway?'
                )
                if not response:
                    self.log('Analysis cancelled by user')
                    return
            
            # Warn if object count is very high (overlap precomputation)
            if n_objects > 500000:
                self.log(f'Warning: {n_objects:,} objects may take 10-60s for overlap analysis...')
                messagebox.showinfo(
                    'Large Dataset',
                    f'Analyzing {n_objects:,} objects.\n'
                    f'Overlap precomputation may take 10-60 seconds.\n'
                    f'Please wait...'
                )
            
            # Get active policy
            active_policy = self.active_policy.get()
            
            self.log(f'Analyzing {n_objects:,} objects using {active_policy} policy...')
            
            # Run empirical analysis based on selected policy
            analyzer = CrowdingAnalyzer(scene)
            
            if active_policy == 'overlap':
                # Only overlap policy
                obj_result = analyzer.analyze_object_policy(m)
                results = {
                    **obj_result,
                    'policy': 'overlap',
                    'discarded_object_ids': obj_result['discarded_object_ids'],
                    'isolated_objects': obj_result['good_targets'],
                    'coverage_fraction': analyzer.compute_coverage_fraction_monte_carlo()
                }
            elif active_policy == 'box':
                # Only box crowding policy
                box_result = analyzer.analyze_box_crowding(a, k0, mode)
                results = {
                    **box_result,
                    'policy': 'box',
                    'discarded_object_ids': box_result['discarded_object_ids'],
                    'isolated_objects': box_result['good_targets'],
                    'coverage_fraction': analyzer.compute_coverage_fraction_monte_carlo()
                }
            else:
                raise ValueError(f"Unknown policy: {active_policy}")
            
            # Compute intensity from data
            H, W = scene.field_shape
            λ_empirical = len(scene) / (H * W)
            self.intensity.set(λ_empirical)
            
            # Build radius distribution from scene
            radii = [obj.radius for obj in scene.objects]
            if len(radii) > 0:
                # Use empirical statistics to inform distribution
                mean_r = np.mean(radii)
                std_r = np.std(radii)
                min_r = max(1.0, min(radii))
                max_r = max(radii)
                
                # Update UI with empirical values
                self.radius_mean.set(float(mean_r))
                self.radius_std.set(float(std_r))
                self.radius_min.set(float(min_r))
                self.radius_max.set(float(max_r))
            
            # Create theoretical model
            R_dist = RadiusDistribution(
                mean=self.radius_mean.get(),
                std=self.radius_std.get(),
                r_min=self.radius_min.get(),
                r_max=self.radius_max.get()
            )
            
            theory = BooleanModelTheory(λ_empirical, R_dist)
            
            # Compute theoretical predictions based on active policy
            coverage_theory = theory.coverage_probability()
            
            if active_policy == 'overlap':
                expected_good = theory.expected_good_count(len(scene), m)
            elif active_policy == 'box':
                # Box policy: approximate as (1 - p_crowded) * n
                p_crowded = theory.box_crowding_prob(a, k0, mode)
                expected_good = len(scene) * (1 - p_crowded)
            else:
                expected_good = 0.0
            
            # Update scoreboard
            self._update_scoreboard(results, coverage_theory, expected_good)
            
            # Update plots
            self._update_plots(theory, results, λ_empirical)
            
            # Store results
            self.last_results = {
                **results,
                'coverage_theory': coverage_theory,
                'expected_good': expected_good,
                'lambda': λ_empirical,
                'active_policy': active_policy,
            }
            
            self.log(f'Analysis complete: {results["isolated_objects"]} isolated objects found using {active_policy} policy')
            
        except Exception as e:
            messagebox.showerror('Analysis Error', f'Failed to run analysis:\n{str(e)}')
            self.log(f'Error: {str(e)}')
            import traceback
            traceback.print_exc()
    
    def _update_scoreboard(
        self,
        results: Dict[str, Any],
        coverage_theory: float,
        expected_good: float
    ) -> None:
        """Update scoreboard labels with results."""
        # Format numbers - using common keys that exist in both policies
        self.metric_labels['total_objects'].config(text=str(results['total_objects']))
        
        # Policy-specific metrics
        policy = results.get('policy', 'overlap')
        if policy == 'box':
            self.metric_labels['crowded_boxes'].config(text=str(results.get('crowded_boxes', 0)))
            self.metric_labels['discarded_by_box'].config(text=str(results.get('discarded_by_box', 0)))
            self.metric_labels['discarded_by_object'].config(text='N/A')
            self.metric_labels['discarded_either'].config(text='N/A')
        else:  # overlap
            self.metric_labels['crowded_boxes'].config(text='N/A')
            self.metric_labels['discarded_by_box'].config(text='N/A')
            self.metric_labels['discarded_by_object'].config(text=str(results.get('discarded_by_object', 0)))
            self.metric_labels['discarded_either'].config(text='N/A')
        
        self.metric_labels['good_targets_strict'].config(text=str(results.get('isolated_objects', 0)))
        self.metric_labels['good_per_area_strict'].config(text=f"{results.get('good_per_area', 0.0):.6f}")
        self.metric_labels['coverage_fraction'].config(text=f"{results['coverage_fraction']:.4f}")
        
        # Theory comparison
        self.metric_labels['coverage_theory'].config(text=f"{coverage_theory:.4f}")
        self.metric_labels['coverage_empirical'].config(text=f"{results['coverage_fraction']:.4f}")
        self.metric_labels['expected_good_object'].config(text=f"{expected_good:.1f}")
        self.metric_labels['actual_good_object'].config(
            text=str(results.get('isolated_objects', 0))
        )
    
    def _update_plots(
        self,
        theory: BooleanModelTheory,
        results: Dict[str, Any],
        λ_current: float
    ) -> None:
        """Update analysis plots."""
        # Clear axes
        self.ax_coverage.clear()
        self.ax_good_targets.clear()
        
        # Plot 1: Coverage vs Intensity
        λ_range = np.linspace(0.001, λ_current * 2, 100)
        coverage_theory_curve = [
            1 - np.exp(-λ * np.pi * theory.R_dist.moment(2))
            for λ in λ_range
        ]
        
        self.ax_coverage.plot(λ_range, coverage_theory_curve, 'b-', label='Theory', linewidth=2)
        self.ax_coverage.plot(
            [λ_current],
            [results['coverage_fraction']],
            'ro',
            markersize=10,
            label='Empirical'
        )
        self.ax_coverage.set_title('Coverage Analysis')
        self.ax_coverage.set_xlabel('Intensity λ (objects per px²)')
        self.ax_coverage.set_ylabel('Coverage Fraction')
        self.ax_coverage.legend()
        self.ax_coverage.grid(True, alpha=0.3)
        
        # Plot 2: Isolated objects vs total count (density-dependent survival)
        n_range = np.linspace(10, max(20.0, results['total_objects'] * 1.5), 100)
        
        # Compute area from current data to vary λ with n correctly
        H, W = self.state.data.geometric_scene.field_shape
        area_px2 = H * W
        
        # Get active policy to show correct theory curve
        policy = results.get('policy', 'overlap')
        
        # Compute theory curve based on active policy
        expected_good_curve = []
        if policy == 'overlap':
            m = self.neighbor_threshold.get()
            for n in n_range:
                λ_n = float(n) / area_px2
                theory_n = BooleanModelTheory(λ=λ_n, R_dist=theory.R_dist)
                p_survive_n = theory_n.palm_survival_probability(m)
                expected_good_curve.append(n * p_survive_n)
            theory_label = 'Theory (Overlap Policy)'
        else:  # box policy
            a = self.box_size.get()
            k0 = self.box_threshold.get()
            mode = self.count_mode.get()
            for n in n_range:
                λ_n = float(n) / area_px2
                theory_n = BooleanModelTheory(λ=λ_n, R_dist=theory.R_dist)
                p_crowded = theory_n.box_crowding_prob(a, k0, mode)
                expected_good_curve.append(n * (1 - p_crowded))
            theory_label = 'Theory (Box Policy)'
        
        self.ax_good_targets.plot(
            n_range,
            expected_good_curve,
            'b-',
            label=theory_label,
            linewidth=2
        )
        self.ax_good_targets.plot(
            [results['total_objects']],
            [results.get('isolated_objects', 0)],
            'ro',
            markersize=10,
            label=f'Empirical ({policy.capitalize()} Policy)'
        )
        self.ax_good_targets.set_title('Isolated Objects vs Density')
        self.ax_good_targets.set_xlabel('Total Objects (n)')
        self.ax_good_targets.set_ylabel('Isolated Objects')
        self.ax_good_targets.legend()
        self.ax_good_targets.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def _update_box_area_display(self) -> None:
        """Update the box area label when slider moves."""
        a = self.box_size.get()
        area = a * a
        self.box_area_label.config(text=f'{area} px²')
    
    def _auto_fit_from_scene(self) -> None:
        """Auto-fit ALL parameters (intensity + radius distribution) from current scene."""
        if not self.state.data.has_geometric_data:
            messagebox.showwarning(
                'No Data',
                'Please generate objects first to auto-fit parameters.'
            )
            return
        
        scene = self.state.data.geometric_scene
        if scene is None or len(scene) == 0:
            messagebox.showwarning('No Data', 'No objects in current scene.')
            return
        
        # Extract radii from scene
        radii = [obj.radius for obj in scene.objects]
        
        if len(radii) == 0:
            messagebox.showwarning('No Data', 'No radii found in objects.')
            return
        
        # Compute radius statistics
        mean_r = float(np.mean(radii))
        std_r = float(np.std(radii))
        min_r = float(max(1.0, min(radii)))
        max_r = float(max(radii))
        
        # Compute intensity (λ) from scene
        H, W = scene.field_shape
        λ_empirical = len(scene) / (H * W)
        
        # Update UI
        self.radius_mean.set(mean_r)
        self.radius_std.set(std_r)
        self.radius_min.set(min_r)
        self.radius_max.set(max_r)
        self.intensity.set(λ_empirical)
        
        # Update status label
        self.fitted_status_label.config(
            text=f'✓ Fitted: λ={λ_empirical:.6f}, μ_R={mean_r:.2f}, σ_R={std_r:.2f}'
        )
        
        self.log(f'Auto-fit complete: λ={λ_empirical:.6f}, μ={mean_r:.2f}, σ={std_r:.2f}')
        messagebox.showinfo(
            'Auto-Fit Complete',
            f'All parameters fitted from scene:\n\n'
            f'Spatial intensity (λ): {λ_empirical:.6f} obj/px²\n'
            f'Radius mean (μ): {mean_r:.2f} px\n'
            f'Radius std (σ): {std_r:.2f} px\n'
            f'Radius range: [{min_r:.2f}, {max_r:.2f}] px'
        )
    
    def _export_csv(self) -> None:
        """Export results to CSV."""
        if self.last_results is None:
            messagebox.showwarning('No Results', 'Run analysis first before exporting.')
            return
        
        from tkinter import filedialog
        import pandas as pd
        
        filepath = filedialog.asksaveasfilename(
            defaultextension='.csv',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
        )
        
        if not filepath:
            return
        
        try:
            # Create DataFrame with results
            df = pd.DataFrame([self.last_results])
            df.to_csv(filepath, index=False)
            self.log(f'Results exported to {filepath}')
            messagebox.showinfo('Export Success', f'Results saved to:\n{filepath}')
        except Exception as e:
            messagebox.showerror('Export Error', f'Failed to export:\n{str(e)}')
    
    def _view_discarded(self) -> None:
        """Open Inspector with discarded objects (sorted appropriately)."""
        if self.last_results is None or self.open_inspector is None:
            messagebox.showwarning('No Results', 'Run analysis first.')
            return
        
        scene = self.state.data.geometric_scene
        policy = self.last_results.get('active_policy', 'overlap')
        
        discarded_ids = list(self.last_results['discarded_object_ids'])
        
        # Sort based on policy
        if policy == 'overlap' and 'neighbor_counts' in self.last_results:
            # Sort by neighbor count (most overlapped first)
            neighbor_counts = self.last_results['neighbor_counts']
            discarded_with_counts = [
                (obj.id, neighbor_counts[i]) 
                for i, obj in enumerate(scene.objects) 
                if obj.id in discarded_ids
            ]
            discarded_with_counts.sort(key=lambda x: x[1], reverse=True)
            discarded_ids_sorted = [obj_id for obj_id, _ in discarded_with_counts]
            sort_desc = 'sorted by overlap count'
        else:
            # Box policy - just use IDs as is
            discarded_ids_sorted = discarded_ids
            sort_desc = ''
        
        self.open_inspector(discarded_ids_sorted)
        self.log(f'Opening inspector with {len(discarded_ids_sorted)} discarded objects ({policy} policy) {sort_desc}')
    
    def _view_good(self) -> None:
        """Open Inspector with isolated objects (sorted appropriately)."""
        if self.last_results is None or self.open_inspector is None:
            messagebox.showwarning('No Results', 'Run analysis first.')
            return
        
        scene = self.state.data.geometric_scene
        if scene is None:
            return
        
        policy = self.last_results.get('active_policy', 'overlap')
        discarded_ids = self.last_results['discarded_object_ids']
        
        # Get isolated object IDs
        isolated_ids = [obj.id for obj in scene.objects if obj.id not in discarded_ids]
        
        # Sort based on policy
        if policy == 'overlap' and 'neighbor_counts' in self.last_results:
            # Sort by neighbor count (ascending - least overlapped first)
            neighbor_counts = self.last_results['neighbor_counts']
            isolated_with_counts = [
                (obj.id, neighbor_counts[i])
                for i, obj in enumerate(scene.objects)
                if obj.id not in discarded_ids
            ]
            isolated_with_counts.sort(key=lambda x: x[1])
            isolated_ids_sorted = [obj_id for obj_id, _ in isolated_with_counts]
            sort_desc = 'sorted by overlap count'
        else:
            # Box policy - just use IDs as is
            isolated_ids_sorted = isolated_ids
            sort_desc = ''
        
        self.open_inspector(isolated_ids_sorted)
        self.log(f'Opening inspector with {len(isolated_ids_sorted)} isolated objects ({policy} policy) {sort_desc}')

