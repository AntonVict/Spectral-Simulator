"""Coverage analysis tab with detailed coverage metrics and plots."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING, Dict, Any, Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from ..statistics_view import StatisticsView


class CoverageTab(ttk.Frame):
    """Simplified coverage analysis showing key metrics and sensitivity plots."""
    
    def __init__(self, parent: tk.Widget, stats_view: StatisticsView):
        """Initialize coverage tab.
        
        Args:
            parent: Parent widget (notebook)
            stats_view: Reference to main statistics view for shared state
        """
        super().__init__(parent)
        self.stats_view = stats_view
        
        # Fix MC samples at 10k (no user control needed)
        self.mc_samples = 10000
        
        # Configure grid layout: Metrics | Plots
        self.columnconfigure(0, weight=1, minsize=300)
        self.columnconfigure(1, weight=2, minsize=600)
        self.rowconfigure(0, weight=1)
        
        # Build UI components
        self._build_metrics()
        self._build_plots()
    
    def _build_metrics(self) -> None:
        """Build coverage metrics and info panel (left column)."""
        metrics_frame = ttk.LabelFrame(self, text='Coverage Analysis')
        metrics_frame.grid(row=0, column=0, sticky='nsew', padx=4, pady=4)
        
        # Create scrollable container
        from ..controls.parameter_widgets import create_scrollable_frame
        canvas, scrollable_frame = create_scrollable_frame(metrics_frame)
        
        # Info section
        info_frame = ttk.LabelFrame(scrollable_frame, text='About Coverage')
        info_frame.pack(fill=tk.X, padx=4, pady=4)
        
        ttk.Label(
            info_frame,
            text='Coverage fraction is the\nproportion of the field\ncovered by at least one\nobject.',
            font=('TkDefaultFont', 8),
            wraplength=250,
            justify=tk.LEFT
        ).pack(padx=4, pady=4)
        
        # Coverage metrics
        metrics_data_frame = ttk.LabelFrame(scrollable_frame, text='Coverage Metrics')
        metrics_data_frame.pack(fill=tk.X, padx=4, pady=8)
        
        coverage_metrics = [
            ('Theoretical', 'theoretical'),
            ('Empirical (MC)', 'empirical_mc'),
        ]
        
        self.metric_labels = {}
        
        for i, (name, key) in enumerate(coverage_metrics):
            ttk.Label(
                metrics_data_frame,
                text=name + ':',
                font=('TkDefaultFont', 9)
            ).grid(row=i, column=0, sticky='w', padx=4, pady=4)
            
            val_label = ttk.Label(
                metrics_data_frame,
                text='—',
                font=('TkDefaultFont', 11, 'bold')
            )
            val_label.grid(row=i, column=1, sticky='e', padx=4, pady=4)
            self.metric_labels[key] = val_label
        
        # Formula section
        formula_frame = ttk.LabelFrame(scrollable_frame, text='Formula')
        formula_frame.pack(fill=tk.X, padx=4, pady=8)
        
        ttk.Label(
            formula_frame,
            text='p = 1 - exp(-λπE[R²])',
            font=('TkDefaultFont', 10, 'bold'),
            foreground='#2a7f2a'
        ).pack(padx=4, pady=(4, 2))
        
        ttk.Label(
            formula_frame,
            text='λ = density\nE[R²] = 2nd moment of radius',
            font=('TkDefaultFont', 8),
            foreground='#555',
            justify=tk.LEFT
        ).pack(padx=4, pady=(0, 4))
    
    def _build_plots(self) -> None:
        """Build coverage sensitivity plots (right column)."""
        plots_frame = ttk.LabelFrame(self, text='Coverage Sensitivity Analysis')
        plots_frame.grid(row=0, column=1, sticky='nsew', padx=4, pady=4)
        
        # Create matplotlib figure with 2 plots (vertical stack)
        self.figure = Figure(figsize=(7, 8), dpi=100)
        self.figure.subplots_adjust(hspace=0.3, left=0.12, right=0.95, top=0.96, bottom=0.06)
        
        # Plot 1: Coverage vs λ (varying intensity)
        self.ax_coverage_lambda = self.figure.add_subplot(211)
        self.ax_coverage_lambda.set_title('Coverage vs Density', fontsize=10)
        self.ax_coverage_lambda.set_xlabel('Density λ', fontsize=9)
        self.ax_coverage_lambda.set_ylabel('Coverage Fraction', fontsize=9)
        self.ax_coverage_lambda.grid(True, alpha=0.3)
        
        # Plot 2: Coverage vs mean radius
        self.ax_coverage_radius = self.figure.add_subplot(212)
        self.ax_coverage_radius.set_title('Coverage vs Mean Radius', fontsize=10)
        self.ax_coverage_radius.set_xlabel('Mean Radius (pixels)', fontsize=9)
        self.ax_coverage_radius.set_ylabel('Coverage Fraction', fontsize=9)
        self.ax_coverage_radius.grid(True, alpha=0.3)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=plots_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
    
    def update_results(self, results: Dict[str, Any], theory, params: Dict[str, Any]) -> None:
        """Update metrics and plots with analysis results.
        
        Args:
            results: Analysis results dictionary
            theory: BooleanModelTheory instance
            params: Analysis parameters
        """
        # Get coverage values
        empirical_mc = results.get('coverage_fraction', 0.0)
        theoretical = theory.coverage_probability() if theory else 0.0
        
        # Update metrics (no error computation)
        self.metric_labels['theoretical'].config(text=f"{theoretical:.4f}")
        self.metric_labels['empirical_mc'].config(text=f"{empirical_mc:.4f}")
        
        # Update plots
        self._update_plots(theory, results, params, empirical_mc, theoretical)
    
    def _update_plots(self, theory, results: Dict[str, Any], params: Dict[str, Any],
                      empirical: float, theoretical: float) -> None:
        """Update coverage sensitivity plots."""
        if not theory:
            return
        
        # Clear plots
        self.ax_coverage_lambda.clear()
        self.ax_coverage_radius.clear()
        
        λ_current = params.get('lambda', 0.001)
        mean_r = params.get('radius_mean', 3.0)
        std_r = params.get('radius_std', 0.6)
        min_r = params.get('radius_min', 2.0)
        max_r = params.get('radius_max', 5.0)
        
        # Plot 1: Coverage vs λ
        λ_values = np.linspace(0, λ_current * 3, 100)
        coverage_values = [1 - np.exp(-λ * np.pi * theory._E_R2) for λ in λ_values]
        
        self.ax_coverage_lambda.plot(λ_values, coverage_values, 'b-', linewidth=2, label='Theoretical')
        self.ax_coverage_lambda.plot(λ_current, empirical, 'ro', markersize=8, label='Empirical (MC)')
        self.ax_coverage_lambda.set_xlabel('Density λ', fontsize=9)
        self.ax_coverage_lambda.set_ylabel('Coverage Fraction', fontsize=9)
        self.ax_coverage_lambda.set_title('Coverage vs Density', fontsize=10)
        self.ax_coverage_lambda.legend(fontsize=8)
        self.ax_coverage_lambda.grid(True, alpha=0.3)
        
        # Plot 2: Coverage vs mean radius (keeping λ fixed)
        from spectral_playground.core.statistics import RadiusDistribution
        
        radius_values = np.linspace(max(0.5, mean_r * 0.3), mean_r * 2, 100)
        coverage_by_radius = []
        
        for r_mean in radius_values:
            # Create radius distribution with proportional std
            r_std_prop = std_r * (r_mean / mean_r)  # Keep ratio constant
            r_min_prop = max(1.0, min_r * (r_mean / mean_r))
            r_max_prop = max_r * (r_mean / mean_r)
            
            R_dist_temp = RadiusDistribution(
                mean=r_mean,
                std=r_std_prop,
                r_min=r_min_prop,
                r_max=r_max_prop
            )
            E_R2_temp = R_dist_temp.moment(2)
            coverage_temp = 1 - np.exp(-λ_current * np.pi * E_R2_temp)
            coverage_by_radius.append(coverage_temp)
        
        self.ax_coverage_radius.plot(radius_values, coverage_by_radius, 'b-', linewidth=2, label='Theoretical')
        self.ax_coverage_radius.plot(mean_r, empirical, 'ro', markersize=8, label='Empirical (MC)')
        self.ax_coverage_radius.set_xlabel('Mean Radius (pixels)', fontsize=9)
        self.ax_coverage_radius.set_ylabel('Coverage Fraction', fontsize=9)
        self.ax_coverage_radius.set_title('Coverage vs Mean Radius', fontsize=10)
        self.ax_coverage_radius.legend(fontsize=8)
        self.ax_coverage_radius.grid(True, alpha=0.3)
        
        # Redraw
        self.canvas.draw()


