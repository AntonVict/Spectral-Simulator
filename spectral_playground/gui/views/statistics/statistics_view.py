"""Main statistics view orchestrator with nested notebook structure."""

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

from .tabs import OverviewTab, CoverageTab, OverlapTab, ParametricTab
from .utils import export_results_to_csv


class StatisticsView(ttk.Frame):
    """Statistical analysis dashboard with nested notebook for organized views.
    
    Provides multiple tabs for different aspects of crowding analysis:
    - Overview: Main controls, key metrics, and primary plots
    - Coverage Analysis: Detailed coverage metrics and theory comparison
    - Overlap Statistics: Distribution analysis and higher-order overlaps
    - Parametric Analysis: Parameter sweeps and sensitivity analysis
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
        
        # Shared analysis results (accessible to all tabs)
        self.last_results: Optional[Dict[str, Any]] = None
        self.last_theory: Optional[BooleanModelTheory] = None
        self.last_params: Optional[Dict[str, Any]] = None
        
        # Create nested notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Create all tabs
        self.overview_tab = OverviewTab(self.notebook, self)
        self.coverage_tab = CoverageTab(self.notebook, self)
        self.overlap_tab = OverlapTab(self.notebook, self)
        self.parametric_tab = ParametricTab(self.notebook, self)
        
        # Add tabs to notebook
        self.notebook.add(self.overview_tab, text='Overview')
        self.notebook.add(self.coverage_tab, text='Coverage Analysis')
        self.notebook.add(self.overlap_tab, text='Overlap Statistics')
        self.notebook.add(self.parametric_tab, text='Parametric Analysis')
        
        # Initialize policy visibility in overview tab
        self.overview_tab.controls.update_policy_visibility()
    
    def run_analysis(self) -> None:
        """Run statistical analysis on current data and update all tabs."""
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
            
            # Get parameters from overview controls
            params = self.overview_tab.controls.get_parameters()
            a = params['box_size']
            k0 = params['box_threshold']
            m = params['neighbor_threshold']
            mode = params['count_mode']
            active_policy = params['active_policy']
            overlap_mode = params['overlap_mode']
            
            # Update overlap mode and clear cached overlap graph if mode changed
            if scene.overlap_mode != overlap_mode:
                self.log(f'Switching overlap detection mode to: {overlap_mode}')
                scene.overlap_mode = overlap_mode
                scene._overlap_graph = None  # Force recomputation with new mode
            
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
            self.overview_tab.controls.intensity.set(λ_empirical)
            
            # Build radius distribution from scene
            radii = [obj.radius for obj in scene.objects]
            if len(radii) > 0:
                mean_r = np.mean(radii)
                std_r = np.std(radii)
                min_r = max(1.0, min(radii))
                max_r = max(radii)
                
                self.overview_tab.controls.radius_mean.set(float(mean_r))
                self.overview_tab.controls.radius_std.set(float(std_r))
                self.overview_tab.controls.radius_min.set(float(min_r))
                self.overview_tab.controls.radius_max.set(float(max_r))
            else:
                # No radii in scene - use current UI values as fallback
                mean_r = params['radius_mean']
                std_r = params['radius_std']
                min_r = params['radius_min']
                max_r = params['radius_max']
            
            # Create theoretical model using computed/updated values
            R_dist = RadiusDistribution(
                mean=mean_r,
                std=std_r,
                r_min=min_r,
                r_max=max_r
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
            
            # Store results for all tabs to access
            results['expected_good'] = expected_good
            params['lambda'] = λ_empirical
            
            self.last_results = results
            self.last_theory = theory
            self.last_params = params
            
            # Update all tabs
            self.overview_tab.update_results(results, theory, params)
            self.coverage_tab.update_results(results, theory, params)
            self.overlap_tab.update_results(results, theory, params)
            # Parametric tab updates on demand when user clicks compute
            
            self.log(f'Analysis complete: {results["isolated_objects"]} isolated objects found using {active_policy} policy')
            
        except Exception as e:
            messagebox.showerror('Analysis Error', f'Failed to run analysis:\n{str(e)}')
            self.log(f'Error: {str(e)}')
            import traceback
            traceback.print_exc()
    
    def export_csv(self) -> None:
        """Export results to CSV."""
        export_results_to_csv(self.last_results, self.log)
    
    def view_discarded(self) -> None:
        """Open Inspector with discarded objects."""
        if self.last_results is None or self.open_inspector is None:
            messagebox.showwarning('No Results', 'Run analysis first.')
            return
        
        scene = self.state.data.geometric_scene
        policy = self.last_results.get('policy', 'overlap')
        
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
    
    def view_good(self) -> None:
        """Open Inspector with isolated objects."""
        if self.last_results is None or self.open_inspector is None:
            messagebox.showwarning('No Results', 'Run analysis first.')
            return
        
        scene = self.state.data.geometric_scene
        if scene is None:
            return
        
        policy = self.last_results.get('policy', 'overlap')
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
