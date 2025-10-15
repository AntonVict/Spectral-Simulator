"""Proximity analysis tab showing near-miss distances and stringency curves."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING, Dict, Any, Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from ..statistics_view import StatisticsView
    from spectral_playground.core.statistics import BooleanModelTheory


class ProximityTab(ttk.Frame):
    """Proximity analysis - measuring near-miss distances and stringency."""
    
    def __init__(self, parent: tk.Widget, stats_view: StatisticsView):
        """Initialize proximity tab.
        
        Args:
            parent: Parent widget (notebook)
            stats_view: Reference to main statistics view for shared state
        """
        super().__init__(parent)
        self.stats_view = stats_view
        
        # Configure grid layout: Info Panel | Plots
        self.columnconfigure(0, weight=1, minsize=300)
        self.columnconfigure(1, weight=2, minsize=600)
        self.rowconfigure(0, weight=1)
        
        # Build UI components
        self._build_info_panel()
        self._build_plots()
    
    def _build_info_panel(self) -> None:
        """Build info and metrics panel (left column)."""
        info_frame = ttk.LabelFrame(self, text='Proximity Metrics')
        info_frame.grid(row=0, column=0, sticky='nsew', padx=4, pady=4)
        
        # Create scrollable container
        from ..controls.parameter_widgets import create_scrollable_frame
        canvas, scrollable_frame = create_scrollable_frame(info_frame)
        
        # Parameters display
        params_frame = ttk.LabelFrame(scrollable_frame, text='Analysis Parameters')
        params_frame.pack(fill=tk.X, padx=4, pady=4)
        
        self.mean_radius_label = ttk.Label(params_frame, text='—', font=('TkDefaultFont', 9))
        self.mean_radius_label.pack(anchor='w', padx=4, pady=2)
        
        self.epsilon_values_label = ttk.Label(params_frame, text='—', font=('TkDefaultFont', 9), wraplength=250)
        self.epsilon_values_label.pack(anchor='w', padx=4, pady=2)
        
        # Metrics display
        metrics_frame = ttk.LabelFrame(scrollable_frame, text='Summary Statistics')
        metrics_frame.pack(fill=tk.X, padx=4, pady=4)
        
        self.total_near_label = ttk.Label(metrics_frame, text='—', font=('TkDefaultFont', 9))
        self.total_near_label.pack(anchor='w', padx=4, pady=2)
        
        self.mean_gap_label = ttk.Label(metrics_frame, text='—', font=('TkDefaultFont', 9))
        self.mean_gap_label.pack(anchor='w', padx=4, pady=2)
        
        self.median_gap_label = ttk.Label(metrics_frame, text='—', font=('TkDefaultFont', 9))
        self.median_gap_label.pack(anchor='w', padx=4, pady=2)
        
        self.objects_with_near_label = ttk.Label(metrics_frame, text='—', font=('TkDefaultFont', 9))
        self.objects_with_near_label.pack(anchor='w', padx=4, pady=2)
        
        # Info text
        info_text_frame = ttk.LabelFrame(scrollable_frame, text='About These Metrics')
        info_text_frame.pack(fill=tk.X, padx=4, pady=4)
        
        info_text = tk.Text(info_text_frame, wrap=tk.WORD, height=12, font=('TkDefaultFont', 9))
        info_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        info_text.insert('1.0', 
            'Gap Distance:\n'
            '  • Distance between object edges\n'
            '  • Positive = not touching\n'
            '  • Measured in pixels\n\n'
            'ε-Neighbors:\n'
            '  • Objects within gap threshold ε\n'
            '  • Includes both overlapping and\n'
            '    near-miss objects\n'
            '  • Shows stringency: how crowded\n'
            '    if we require X pixels gap?\n\n'
            'Epsilon values auto-set to:\n'
            '  [0, 0.5×r̄, r̄, 2×r̄, 5×r̄]\n'
            '  where r̄ = mean radius'
        )
        info_text.config(state=tk.DISABLED)
    
    def _build_plots(self) -> None:
        """Build plots panel (right column)."""
        plots_frame = ttk.Frame(self)
        plots_frame.grid(row=0, column=1, sticky='nsew', padx=4, pady=4)
        plots_frame.rowconfigure(0, weight=1)
        plots_frame.rowconfigure(1, weight=1)
        plots_frame.columnconfigure(0, weight=1)
        
        # Plot 1: ε-Neighbor Count Curve
        plot1_frame = ttk.LabelFrame(plots_frame, text='ε-Neighbor Count (Stringency Analysis)')
        plot1_frame.grid(row=0, column=0, sticky='nsew', padx=2, pady=2)
        
        self.fig1 = Figure(figsize=(6, 3), dpi=100)
        self.ax1 = self.fig1.add_subplot(111)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=plot1_frame)
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Plot 2: Gap Distance Histogram
        plot2_frame = ttk.LabelFrame(plots_frame, text='Gap Distance Distribution (Near-Misses)')
        plot2_frame.grid(row=1, column=0, sticky='nsew', padx=2, pady=2)
        
        self.fig2 = Figure(figsize=(6, 3), dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=plot2_frame)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial empty plots
        self._clear_plots()
    
    def _clear_plots(self) -> None:
        """Clear both plots."""
        self.ax1.clear()
        self.ax1.set_xlabel('Gap Threshold ε (pixels)')
        self.ax1.set_ylabel('Mean Number of ε-Neighbors')
        self.ax1.set_title('No data')
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.clear()
        self.ax2.set_xlabel('Gap Distance (pixels)')
        self.ax2.set_ylabel('Number of Pairs')
        self.ax2.set_title('No data')
        self.ax2.grid(True, alpha=0.3)
        
        self.canvas1.draw_idle()
        self.canvas2.draw_idle()
    
    def update_results(self, results: Dict[str, Any], theory: Optional[BooleanModelTheory]) -> None:
        """Update tab with new analysis results.
        
        Args:
            results: Analysis results from CrowdingAnalyzer
            theory: Theoretical model (not used yet)
        """
        # Check if we have geometric scene
        if not self.stats_view.state.data.has_geometric_data:
            self._clear_plots()
            self._update_metrics(None, None, None)
            return
        
        scene = self.stats_view.state.data.geometric_scene
        
        # Compute mean radius
        radii = np.array([obj.radius for obj in scene.objects])
        if len(radii) == 0:
            self._clear_plots()
            self._update_metrics(None, None, None)
            return
        
        mean_radius = np.mean(radii)
        
        # Generate epsilon values: [0, 0.5×r̄, r̄, 2×r̄, 5×r̄]
        epsilon_values = [0, 0.5 * mean_radius, mean_radius, 2 * mean_radius, 5 * mean_radius]
        
        # Compute proximity metrics (for gap < mean_radius)
        try:
            proximity_metrics = scene.compute_proximity_metrics(epsilon=mean_radius)
            epsilon_curves = scene.compute_epsilon_neighbor_curves(epsilon_values)
        except Exception as e:
            self.stats_view.log(f"Error computing proximity metrics: {e}")
            self._clear_plots()
            self._update_metrics(None, None, None)
            return
        
        # Update displays
        self._update_metrics(proximity_metrics, epsilon_curves, mean_radius)
        self._update_plots(proximity_metrics, epsilon_curves, mean_radius)
    
    def _update_metrics(
        self, 
        proximity_metrics: Optional[Dict[str, Any]], 
        epsilon_curves: Optional[Dict[str, Any]],
        mean_radius: Optional[float]
    ) -> None:
        """Update metrics labels.
        
        Args:
            proximity_metrics: Results from compute_proximity_metrics() or None
            epsilon_curves: Results from compute_epsilon_neighbor_curves() or None
            mean_radius: Mean object radius or None
        """
        if proximity_metrics is None or epsilon_curves is None or mean_radius is None:
            self.mean_radius_label.config(text='Mean radius: —')
            self.epsilon_values_label.config(text='Epsilon values: —')
            self.total_near_label.config(text='Total near-miss pairs: —')
            self.mean_gap_label.config(text='Mean gap distance: —')
            self.median_gap_label.config(text='Median gap distance: —')
            self.objects_with_near_label.config(text='Objects with near-neighbors: —')
            return
        
        pairs = proximity_metrics['proximity_pairs']
        per_object_near_count = proximity_metrics['per_object_near_count']
        epsilon_vals = epsilon_curves['epsilon_vals']
        
        # Format epsilon values
        eps_str = ', '.join([f'{e:.1f}' for e in epsilon_vals])
        
        # Update parameter labels
        self.mean_radius_label.config(text=f'Mean radius: {mean_radius:.2f} pixels')
        self.epsilon_values_label.config(text=f'ε values: [{eps_str}] px')
        
        if not pairs:
            self.total_near_label.config(text=f'Total near-miss pairs (gap < {mean_radius:.1f}px): 0')
            self.mean_gap_label.config(text='Mean gap distance: —')
            self.median_gap_label.config(text='Median gap distance: —')
            self.objects_with_near_label.config(text='Objects with near-neighbors: 0 (0%)')
            return
        
        # Extract gap distances
        gaps = [p[2] for p in pairs]
        mean_gap = np.mean(gaps)
        median_gap = np.median(gaps)
        
        # Count objects with near-neighbors
        n_objects_with_near = np.sum(per_object_near_count > 0)
        n_total_objects = len(per_object_near_count)
        pct_with_near = 100 * n_objects_with_near / n_total_objects if n_total_objects > 0 else 0
        
        # Update labels
        self.total_near_label.config(text=f'Total near-miss pairs (gap < {mean_radius:.1f}px): {len(pairs)}')
        self.mean_gap_label.config(text=f'Mean gap distance: {mean_gap:.2f} pixels')
        self.median_gap_label.config(text=f'Median gap distance: {median_gap:.2f} pixels')
        self.objects_with_near_label.config(
            text=f'Objects with near-neighbors: {n_objects_with_near} ({pct_with_near:.1f}%)'
        )
    
    def _update_plots(
        self, 
        proximity_metrics: Dict[str, Any], 
        epsilon_curves: Dict[str, Any],
        mean_radius: float
    ) -> None:
        """Update both plots.
        
        Args:
            proximity_metrics: Results from compute_proximity_metrics()
            epsilon_curves: Results from compute_epsilon_neighbor_curves()
            mean_radius: Mean object radius
        """
        pairs = proximity_metrics['proximity_pairs']
        epsilon_vals = epsilon_curves['epsilon_vals']
        mean_neighbors = epsilon_curves['mean_neighbors']
        
        # Plot 1: ε-Neighbor Count Curve
        self.ax1.clear()
        
        if epsilon_vals and mean_neighbors:
            self.ax1.plot(epsilon_vals, mean_neighbors, 'o-', linewidth=2, 
                         markersize=8, color='#2a7f2a', label='Mean neighbors')
            
            self.ax1.set_xlabel('Gap Threshold ε (pixels)', fontsize=10)
            self.ax1.set_ylabel('Mean Number of ε-Neighbors', fontsize=10)
            self.ax1.set_title('Neighbor Count vs Gap Tolerance (Stringency)', fontsize=10)
            self.ax1.grid(True, alpha=0.3)
            self.ax1.legend(fontsize=9)
            
            # Add vertical line at mean_radius
            self.ax1.axvline(mean_radius, color='gray', linestyle='--', 
                            alpha=0.5, label=f'ε = r̄ ({mean_radius:.1f}px)')
        else:
            self.ax1.text(0.5, 0.5, 'No data', 
                         ha='center', va='center', transform=self.ax1.transAxes)
            self.ax1.set_xlabel('Gap Threshold ε (pixels)', fontsize=10)
            self.ax1.set_ylabel('Mean Number of ε-Neighbors', fontsize=10)
        
        self.fig1.tight_layout()
        
        # Plot 2: Gap Distance Histogram
        self.ax2.clear()
        
        if pairs:
            gaps = [p[2] for p in pairs]
            max_epsilon = max(epsilon_vals) if epsilon_vals else mean_radius
            
            # Create bins from 0 to max_epsilon
            n_bins = min(20, len(gaps) // 5 + 1)  # At least 5 samples per bin
            n_bins = max(10, n_bins)  # At least 10 bins
            bins = np.linspace(0, max_epsilon, n_bins)
            
            self.ax2.hist(gaps, bins=bins, edgecolor='black', linewidth=0.5, 
                         color='#2a7f7f', alpha=0.7)
            
            self.ax2.set_xlabel('Gap Distance (pixels)', fontsize=10)
            self.ax2.set_ylabel('Number of Pairs', fontsize=10)
            self.ax2.set_title(f'Gap Distance Distribution (n={len(gaps)} near-misses)', fontsize=10)
            self.ax2.grid(True, alpha=0.3, axis='y')
            
            # Add vertical line at mean gap
            mean_gap = np.mean(gaps)
            self.ax2.axvline(mean_gap, color='red', linestyle='--', 
                            linewidth=2, label=f'Mean: {mean_gap:.1f}px')
            self.ax2.legend(fontsize=9)
        else:
            self.ax2.text(0.5, 0.5, 'No near-miss pairs', 
                         ha='center', va='center', transform=self.ax2.transAxes)
            self.ax2.set_xlabel('Gap Distance (pixels)', fontsize=10)
            self.ax2.set_ylabel('Number of Pairs', fontsize=10)
        
        self.fig2.tight_layout()
        
        # Redraw canvases
        self.canvas1.draw_idle()
        self.canvas2.draw_idle()

