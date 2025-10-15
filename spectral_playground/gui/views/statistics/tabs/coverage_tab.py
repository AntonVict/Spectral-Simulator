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
    """Detailed coverage analysis with theory vs empirical comparison."""
    
    def __init__(self, parent: tk.Widget, stats_view: StatisticsView):
        """Initialize coverage tab.
        
        Args:
            parent: Parent widget (notebook)
            stats_view: Reference to main statistics view for shared state
        """
        super().__init__(parent)
        self.stats_view = stats_view
        
        # Configure grid layout: Controls | Metrics | Plots
        self.columnconfigure(0, weight=1, minsize=200)
        self.columnconfigure(1, weight=1, minsize=250)
        self.columnconfigure(2, weight=2, minsize=500)
        self.rowconfigure(0, weight=1)
        
        # Build UI components
        self._build_controls()
        self._build_metrics()
        self._build_plots()
    
    def _build_controls(self) -> None:
        """Build coverage-specific controls (left column)."""
        controls_frame = ttk.LabelFrame(self, text='Coverage Controls')
        controls_frame.grid(row=0, column=0, sticky='nsew', padx=4, pady=4)
        
        # Monte Carlo samples
        ttk.Label(controls_frame, text='Monte Carlo Samples:').pack(anchor='w', padx=4, pady=(8, 2))
        
        self.mc_samples = tk.IntVar(value=10000)
        
        ttk.Scale(
            controls_frame,
            from_=1000,
            to=50000,
            variable=self.mc_samples,
            orient=tk.HORIZONTAL
        ).pack(fill=tk.X, padx=4)
        
        samples_label = ttk.Label(controls_frame, textvariable=self.mc_samples)
        samples_label.pack(anchor='w', padx=4)
        
        ttk.Label(
            controls_frame,
            text='Higher = more accurate\nbut slower',
            font=('TkDefaultFont', 8, 'italic'),
            foreground='#555'
        ).pack(anchor='w', padx=4, pady=(0, 8))
        
        # Computation method
        ttk.Label(controls_frame, text='Computation Method:').pack(anchor='w', padx=4, pady=(8, 2))
        
        self.use_exact = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(
            controls_frame,
            text='Use exact rasterization',
            variable=self.use_exact
        ).pack(anchor='w', padx=4)
        
        ttk.Label(
            controls_frame,
            text='⚠ Warning: Exact method\ncan be very slow for\nlarge datasets',
            font=('TkDefaultFont', 8, 'italic'),
            foreground='#cc0000'
        ).pack(anchor='w', padx=4, pady=(2, 8))
        
        # Info section
        info_frame = ttk.LabelFrame(controls_frame, text='About Coverage')
        info_frame.pack(fill=tk.X, padx=4, pady=8)
        
        ttk.Label(
            info_frame,
            text='Coverage fraction is the\nproportion of the field\ncovered by at least one\nobject.',
            font=('TkDefaultFont', 8),
            wraplength=180,
            justify=tk.LEFT
        ).pack(padx=4, pady=4)
    
    def _build_metrics(self) -> None:
        """Build coverage metrics table (middle column)."""
        metrics_frame = ttk.LabelFrame(self, text='Coverage Metrics')
        metrics_frame.grid(row=0, column=1, sticky='nsew', padx=4, pady=4)
        
        # Create scrollable container
        from ..controls.parameter_widgets import create_scrollable_frame
        canvas, scrollable_frame = create_scrollable_frame(metrics_frame)
        
        # Coverage metrics
        coverage_metrics = [
            ('Empirical (Monte Carlo)', 'empirical_mc'),
            ('Theoretical (Formula)', 'theoretical'),
            ('', ''),  # Separator
            ('Absolute Error', 'abs_error'),
            ('Relative Error (%)', 'rel_error'),
        ]
        
        self.metric_labels = {}
        row = 0
        
        for name, key in coverage_metrics:
            if not name:
                ttk.Separator(scrollable_frame, orient='horizontal').grid(
                    row=row, column=0, columnspan=2, sticky='ew', pady=4
                )
                row += 1
                continue
            
            ttk.Label(
                scrollable_frame,
                text=name + ':',
                font=('TkDefaultFont', 9)
            ).grid(row=row, column=0, sticky='w', padx=4, pady=2)
            
            val_label = ttk.Label(
                scrollable_frame,
                text='—',
                font=('TkDefaultFont', 11, 'bold')
            )
            val_label.grid(row=row, column=1, sticky='e', padx=4, pady=2)
            
            if key:
                self.metric_labels[key] = val_label
            row += 1
        
        # Additional info
        info_frame = ttk.LabelFrame(scrollable_frame, text='Formula')
        info_frame.grid(row=row, column=0, columnspan=2, sticky='ew', padx=4, pady=8)
        
        ttk.Label(
            info_frame,
            text='p = 1 - exp(-λπE[R²])',
            font=('TkDefaultFont', 10, 'bold'),
            foreground='#2a7f2a'
        ).pack(padx=4, pady=4)
        
        ttk.Label(
            info_frame,
            text='where λ = density,\nE[R²] = 2nd moment of radius',
            font=('TkDefaultFont', 8),
            justify=tk.LEFT
        ).pack(padx=4, pady=(0, 4))
    
    def _build_plots(self) -> None:
        """Build coverage plots (right column)."""
        plots_frame = ttk.LabelFrame(self, text='Coverage Plots')
        plots_frame.grid(row=0, column=2, sticky='nsew', padx=4, pady=4)
        
        # Create matplotlib figure with 2x2 grid
        self.figure = Figure(figsize=(7, 8), dpi=100)
        self.figure.subplots_adjust(hspace=0.35, wspace=0.3, left=0.12, right=0.95, top=0.96, bottom=0.06)
        
        # Plot 1: Coverage vs λ (varying intensity)
        self.ax_coverage_lambda = self.figure.add_subplot(221)
        self.ax_coverage_lambda.set_title('Coverage vs Density', fontsize=10)
        self.ax_coverage_lambda.set_xlabel('Density λ', fontsize=9)
        self.ax_coverage_lambda.set_ylabel('Coverage Fraction', fontsize=9)
        self.ax_coverage_lambda.grid(True, alpha=0.3)
        
        # Plot 2: Coverage vs mean radius
        self.ax_coverage_radius = self.figure.add_subplot(222)
        self.ax_coverage_radius.set_title('Coverage vs Mean Radius', fontsize=10)
        self.ax_coverage_radius.set_xlabel('Mean Radius (pixels)', fontsize=9)
        self.ax_coverage_radius.set_ylabel('Coverage Fraction', fontsize=9)
        self.ax_coverage_radius.grid(True, alpha=0.3)
        
        # Plot 3: Theory-empirical comparison
        self.ax_comparison = self.figure.add_subplot(223)
        self.ax_comparison.set_title('Theory vs Empirical', fontsize=10)
        self.ax_comparison.set_ylabel('Coverage Fraction', fontsize=9)
        
        # Plot 4: Coverage vs E[R²]
        self.ax_coverage_r2 = self.figure.add_subplot(224)
        self.ax_coverage_r2.set_title('Coverage vs E[R²]', fontsize=10)
        self.ax_coverage_r2.set_xlabel('E[R²] (pixels²)', fontsize=9)
        self.ax_coverage_r2.set_ylabel('Coverage Fraction', fontsize=9)
        self.ax_coverage_r2.grid(True, alpha=0.3)
        
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
        
        # Compute errors
        abs_error = abs(empirical_mc - theoretical)
        rel_error = (abs_error / theoretical * 100) if theoretical > 0 else 0.0
        
        # Update metrics
        self.metric_labels['empirical_mc'].config(text=f"{empirical_mc:.4f}")
        self.metric_labels['theoretical'].config(text=f"{theoretical:.4f}")
        self.metric_labels['abs_error'].config(text=f"{abs_error:.4f}")
        self.metric_labels['rel_error'].config(text=f"{rel_error:.2f}%")
        
        # Update plots
        self._update_plots(theory, results, params, empirical_mc, theoretical)
    
    def _update_plots(self, theory, results: Dict[str, Any], params: Dict[str, Any],
                      empirical: float, theoretical: float) -> None:
        """Update coverage plots."""
        if not theory:
            return
        
        # Clear all plots
        self.ax_coverage_lambda.clear()
        self.ax_coverage_radius.clear()
        self.ax_comparison.clear()
        self.ax_coverage_r2.clear()
        
        λ_current = params.get('lambda', 0.001)
        mean_r = params.get('radius_mean', 3.0)
        std_r = params.get('radius_std', 0.6)
        min_r = params.get('radius_min', 2.0)
        max_r = params.get('radius_max', 5.0)
        
        # Plot 1: Coverage vs λ
        λ_values = np.linspace(0, λ_current * 3, 100)
        coverage_values = [1 - np.exp(-λ * np.pi * theory._E_R2) for λ in λ_values]
        
        self.ax_coverage_lambda.plot(λ_values, coverage_values, 'b-', linewidth=2, label='Theory')
        self.ax_coverage_lambda.plot(λ_current, theoretical, 'ro', markersize=8, label='Current')
        self.ax_coverage_lambda.plot(λ_current, empirical, 'g^', markersize=8, label='Empirical')
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
        
        self.ax_coverage_radius.plot(radius_values, coverage_by_radius, 'b-', linewidth=2, label='Theory')
        self.ax_coverage_radius.plot(mean_r, theoretical, 'ro', markersize=8, label='Current')
        self.ax_coverage_radius.plot(mean_r, empirical, 'g^', markersize=8, label='Empirical')
        self.ax_coverage_radius.set_xlabel('Mean Radius (pixels)', fontsize=9)
        self.ax_coverage_radius.set_ylabel('Coverage Fraction', fontsize=9)
        self.ax_coverage_radius.set_title('Coverage vs Mean Radius', fontsize=10)
        self.ax_coverage_radius.legend(fontsize=8)
        self.ax_coverage_radius.grid(True, alpha=0.3)
        
        # Plot 3: Theory vs Empirical comparison (bar chart)
        categories = ['Theoretical', 'Empirical (MC)']
        values = [theoretical, empirical]
        colors = ['#4a90e2', '#50c878']
        
        bars = self.ax_comparison.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        self.ax_comparison.set_ylabel('Coverage Fraction', fontsize=9)
        self.ax_comparison.set_title('Theory vs Empirical', fontsize=10)
        self.ax_comparison.set_ylim(0, min(1.0, max(values) * 1.2))
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            self.ax_comparison.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{val:.4f}',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )
        
        # Plot 4: Coverage vs E[R²]
        E_R2_values = np.linspace(0, theory._E_R2 * 3, 100)
        coverage_by_r2 = [1 - np.exp(-λ_current * np.pi * E_R2) for E_R2 in E_R2_values]
        
        self.ax_coverage_r2.plot(E_R2_values, coverage_by_r2, 'b-', linewidth=2, label='Theory')
        self.ax_coverage_r2.plot(theory._E_R2, theoretical, 'ro', markersize=8, label='Current')
        self.ax_coverage_r2.plot(theory._E_R2, empirical, 'g^', markersize=8, label='Empirical')
        self.ax_coverage_r2.set_xlabel('E[R²] (pixels²)', fontsize=9)
        self.ax_coverage_r2.set_ylabel('Coverage Fraction', fontsize=9)
        self.ax_coverage_r2.set_title('Coverage vs E[R²]', fontsize=10)
        self.ax_coverage_r2.legend(fontsize=8)
        self.ax_coverage_r2.grid(True, alpha=0.3)
        
        # Redraw
        self.canvas.draw()


