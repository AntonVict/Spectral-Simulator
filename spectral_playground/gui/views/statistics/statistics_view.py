"""Main statistics view orchestrator."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, Optional, Dict, Any

import numpy as np

from spectral_playground.gui.state import PlaygroundState
from spectral_playground.core.statistics import (
    RadiusDistribution,
    BooleanModelTheory,
    CrowdingAnalyzer
)

from .controls import ControlsPanel
from .metrics import MetricsPanel
from .plots import PlotsPanel
from .plots.plot_updater import PlotUpdater
from .utils import export_results_to_csv


class StatisticsView(ttk.Frame):
    """Statistical analysis dashboard with numbers-first reporting.
    
    Orchestrates controls, metrics display, and plots for crowding analysis.
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
        
        # Configure grid layout: Controls | Metrics | Plots
        self.columnconfigure(0, weight=1, minsize=250)
        self.columnconfigure(1, weight=1, minsize=280)
        self.columnconfigure(2, weight=2, minsize=500)
        self.rowconfigure(0, weight=1)
        
        # Create panels
        self.controls = ControlsPanel(
            self,
            state,
            log_callback,
            run_analysis_callback=self._run_analysis,
            export_callback=self._export_csv
        )
        self.controls.grid(row=0, column=0, sticky='nsew', padx=4, pady=4)
        
        self.metrics = MetricsPanel(self)
        self.metrics.grid(row=0, column=1, sticky='nsew', padx=4, pady=4)
        self.metrics.set_inspector_callbacks(
            view_discarded_cb=self._view_discarded,
            view_good_cb=self._view_good
        )
        
        plots_frame = ttk.LabelFrame(self, text='Analysis Plots')
        plots_frame.grid(row=0, column=2, sticky='nsew', padx=4, pady=4)
        self.plots = PlotsPanel(plots_frame)
        self.plots.pack(fill=tk.BOTH, expand=True)
        
        # Initialize policy visibility
        self.controls.update_policy_visibility()
        self.metrics.update_policy_visibility(self.controls.active_policy.get())
        
        # Connect policy changes to metrics
        self.controls.active_policy.trace_add(
            'write',
            lambda *args: self.metrics.update_policy_visibility(self.controls.active_policy.get())
        )
    
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
            
            # Get parameters from controls
            params = self.controls.get_parameters()
            a = params['box_size']
            k0 = params['box_threshold']
            m = params['neighbor_threshold']
            mode = params['count_mode']
            active_policy = params['active_policy']
            
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
            
            # Warn if object count is very high
            if n_objects > 500000:
                self.log(f'Warning: {n_objects:,} objects may take 10-60s for overlap analysis...')
                messagebox.showinfo(
                    'Large Dataset',
                    f'Analyzing {n_objects:,} objects.\n'
                    f'Overlap precomputation may take 10-60 seconds.\n'
                    f'Please wait...'
                )
            
            self.log(f'Analyzing {n_objects:,} objects using {active_policy} policy...')
            
            # Run empirical analysis
            analyzer = CrowdingAnalyzer(scene)
            
            if active_policy == 'overlap':
                obj_result = analyzer.analyze_object_policy(m)
                results = {
                    **obj_result,
                    'policy': 'overlap',
                    'discarded_object_ids': obj_result['discarded_object_ids'],
                    'isolated_objects': obj_result['good_targets'],
                    'coverage_fraction': analyzer.compute_coverage_fraction_monte_carlo()
                }
            elif active_policy == 'box':
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
            λ_empirical = len(scene) / (H * W)
            self.controls.intensity.set(λ_empirical)
            
            # Build radius distribution from scene
            radii = [obj.radius for obj in scene.objects]
            if len(radii) > 0:
                mean_r = np.mean(radii)
                std_r = np.std(radii)
                min_r = max(1.0, min(radii))
                max_r = max(radii)
                
                self.controls.radius_mean.set(float(mean_r))
                self.controls.radius_std.set(float(std_r))
                self.controls.radius_min.set(float(min_r))
                self.controls.radius_max.set(float(max_r))
            else:
                # No radii in scene - use current UI values as fallback
                mean_r = params['radius_mean']
                std_r = params['radius_std']
                min_r = params['radius_min']
                max_r = params['radius_max']
            
            # Create theoretical model using computed/updated values
            R_dist = RadiusDistribution(
                mean=mean_r,      # Use computed value, not stale params
                std=std_r,        # Use computed value, not stale params
                r_min=min_r,      # Use computed value, not stale params
                r_max=max_r       # Use computed value, not stale params
            )
            
            theory = BooleanModelTheory(λ_empirical, R_dist)
            
            # Compute theoretical predictions
            if active_policy == 'overlap':
                expected_good = theory.expected_good_count(len(scene), m)
            elif active_policy == 'box':
                p_crowded = theory.box_crowding_prob(a, k0, mode)
                expected_good = len(scene) * (1 - p_crowded)
            else:
                expected_good = 0.0
            
            # Update metrics
            self.metrics.update_metrics(results, expected_good)
            
            # Update plots
            self._update_plots(theory, results, λ_empirical, params)
            
            # Store results
            self.last_results = {
                **results,
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
    
    def _update_plots(
        self,
        theory: BooleanModelTheory,
        results: Dict[str, Any],
        λ_current: float,
        params: Dict[str, Any]
    ) -> None:
        """Update all analysis plots.
        
        Args:
            theory: Boolean model theory instance
            results: Analysis results
            λ_current: Current intensity
            params: Analysis parameters
        """
        # Clear all plots
        self.plots.clear_all()
        
        policy = results.get('policy', 'overlap')
        
        # Update survival vs lambda plot
        PlotUpdater.update_survival_vs_lambda(
            self.plots.ax_survival_lambda,
            theory,
            results,
            λ_current,
            policy,
            params
        )
        
        # Update survival vs threshold plot
        PlotUpdater.update_survival_vs_threshold(
            self.plots.ax_survival_threshold,
            theory,
            results,
            λ_current,
            policy,
            params
        )
        
        # Update isolated count plot
        H, W = self.state.data.geometric_scene.field_shape
        area_px2 = H * W
        
        PlotUpdater.update_isolated_count(
            self.plots.ax_isolated_count,
            theory,
            results,
            area_px2,
            policy,
            params
        )
        
        # Redraw
        self.plots.draw()
    
    def _export_csv(self) -> None:
        """Export results to CSV."""
        export_results_to_csv(self.last_results, self.log)
    
    def _view_discarded(self) -> None:
        """Open Inspector with discarded objects."""
        if self.last_results is None or self.open_inspector is None:
            messagebox.showwarning('No Results', 'Run analysis first.')
            return
        
        scene = self.state.data.geometric_scene
        policy = self.last_results.get('active_policy', 'overlap')
        
        discarded_ids = list(self.last_results['discarded_object_ids'])
        
        # Sort based on policy
        if policy == 'overlap' and 'neighbor_counts' in self.last_results:
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
            discarded_ids_sorted = discarded_ids
            sort_desc = ''
        
        self.open_inspector(discarded_ids_sorted)
        self.log(f'Opening inspector with {len(discarded_ids_sorted)} discarded objects ({policy} policy) {sort_desc}')
    
    def _view_good(self) -> None:
        """Open Inspector with isolated objects."""
        if self.last_results is None or self.open_inspector is None:
            messagebox.showwarning('No Results', 'Run analysis first.')
            return
        
        scene = self.state.data.geometric_scene
        if scene is None:
            return
        
        policy = self.last_results.get('active_policy', 'overlap')
        discarded_ids = self.last_results['discarded_object_ids']
        
        isolated_ids = [obj.id for obj in scene.objects if obj.id not in discarded_ids]
        
        # Sort based on policy
        if policy == 'overlap' and 'neighbor_counts' in self.last_results:
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
            isolated_ids_sorted = isolated_ids
            sort_desc = ''
        
        self.open_inspector(isolated_ids_sorted)
        self.log(f'Opening inspector with {len(isolated_ids_sorted)} isolated objects ({policy} policy) {sort_desc}')

