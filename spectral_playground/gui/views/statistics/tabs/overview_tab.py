"""Overview tab showing streamlined metrics and key plots."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING, Dict, Any, Optional

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from ..statistics_view import StatisticsView


class OverviewTab(ttk.Frame):
    """Streamlined overview with controls, key metrics, and main plots."""
    
    def __init__(self, parent: tk.Widget, stats_view: StatisticsView):
        """Initialize overview tab.
        
        Args:
            parent: Parent widget (notebook)
            stats_view: Reference to main statistics view for shared state
        """
        super().__init__(parent)
        self.stats_view = stats_view
        
        # Configure grid layout: Controls | Metrics | Plots
        self.columnconfigure(0, weight=1, minsize=250)
        self.columnconfigure(1, weight=1, minsize=280)
        self.columnconfigure(2, weight=2, minsize=500)
        self.rowconfigure(0, weight=1)
        
        # Build UI components
        self._build_controls()
        self._build_metrics()
        self._build_plots()
    
    def _build_controls(self) -> None:
        """Build the controls panel (left column)."""
        # Import here to avoid circular imports
        from ..controls import ControlsPanel
        
        self.controls = ControlsPanel(
            self,
            self.stats_view.state,
            self.stats_view.log,
            run_analysis_callback=self.stats_view.run_analysis,
            export_callback=self.stats_view.export_csv
        )
        self.controls.grid(row=0, column=0, sticky='nsew', padx=4, pady=4)
    
    def _build_metrics(self) -> None:
        """Build the key metrics panel (middle column)."""
        metrics_frame = ttk.LabelFrame(self, text='Key Metrics')
        metrics_frame.grid(row=0, column=1, sticky='nsew', padx=4, pady=4)
        
        # Create scrollable container
        from ..controls.parameter_widgets import create_scrollable_frame
        canvas, scrollable_frame = create_scrollable_frame(metrics_frame)
        
        # Define key metrics to show
        key_metrics = [
            ('Total Objects', 'total_objects'),
            ('', ''),  # Separator
            ('Isolated Objects', 'isolated_objects'),
            ('Isolated/Area (per px²)', 'good_per_area'),
            ('', ''),  # Separator
            ('Coverage Fraction', 'coverage_fraction'),
        ]
        
        self.metric_labels = {}
        row = 0
        
        for name, key in key_metrics:
            if not name:
                ttk.Separator(scrollable_frame, orient='horizontal').grid(
                    row=row, column=0, columnspan=2, sticky='ew', pady=4
                )
                row += 1
                continue
            
            ttk.Label(
                scrollable_frame,
                text=name + ':',
                font=('TkDefaultFont', 10)
            ).grid(row=row, column=0, sticky='w', padx=4, pady=2)
            
            val_label = ttk.Label(
                scrollable_frame,
                text='—',
                font=('TkDefaultFont', 12, 'bold')
            )
            val_label.grid(row=row, column=1, sticky='e', padx=4, pady=2)
            
            if key:
                self.metric_labels[key] = val_label
            row += 1
        
        # Theory comparison section
        theory_frame = ttk.LabelFrame(scrollable_frame, text='Theory vs Empirical')
        theory_frame.grid(row=row, column=0, columnspan=2, sticky='ew', padx=4, pady=8)
        
        theory_metrics = [
            ('Expected Isolated', 'expected_good'),
            ('Actual Isolated', 'actual_good'),
        ]
        
        for i, (name, key) in enumerate(theory_metrics):
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
        
        row += 1
        
        # Navigation buttons
        nav_frame = ttk.LabelFrame(scrollable_frame, text='Detailed Analysis')
        nav_frame.grid(row=row, column=0, columnspan=2, sticky='ew', padx=4, pady=8)
        
        ttk.Button(
            nav_frame,
            text='See Coverage Details →',
            command=lambda: self.stats_view.notebook.select(1)
        ).pack(fill=tk.X, padx=4, pady=2)
        
        ttk.Button(
            nav_frame,
            text='See Neighbor Analysis →',
            command=lambda: self.stats_view.notebook.select(2)
        ).pack(fill=tk.X, padx=4, pady=2)
        
        ttk.Button(
            nav_frame,
            text='See Parametric Analysis →',
            command=lambda: self.stats_view.notebook.select(3)
        ).pack(fill=tk.X, padx=4, pady=2)
    
    def _build_plots(self) -> None:
        """Build the plots panel (right column)."""
        plots_frame = ttk.LabelFrame(self, text='Analysis Plots')
        plots_frame.grid(row=0, column=2, sticky='nsew', padx=4, pady=4)
        
        # Create matplotlib figure with 2 subplots
        self.figure = Figure(figsize=(7, 8), dpi=100)
        self.figure.subplots_adjust(hspace=0.3, left=0.13, right=0.95, top=0.96, bottom=0.06)
        
        # Plot 1: Survival probability vs intensity
        self.ax_survival_lambda = self.figure.add_subplot(211)
        self.ax_survival_lambda.set_title('Probability of Isolation vs Object Density')
        self.ax_survival_lambda.set_xlabel('Density λ (objects per px²)')
        self.ax_survival_lambda.set_ylabel('P(isolated)')
        self.ax_survival_lambda.grid(True, alpha=0.3)
        
        # Plot 2: Isolated objects vs density
        self.ax_isolated_count = self.figure.add_subplot(212)
        self.ax_isolated_count.set_title('Number of Isolated Objects vs Density')
        self.ax_isolated_count.set_xlabel('Total Objects (n)')
        self.ax_isolated_count.set_ylabel('Number of Isolated Objects')
        self.ax_isolated_count.grid(True, alpha=0.3)
        
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
        # Update metrics
        self.metric_labels['total_objects'].config(text=str(results['total_objects']))
        self.metric_labels['isolated_objects'].config(text=str(results.get('isolated_objects', 0)))
        self.metric_labels['good_per_area'].config(text=f"{results.get('good_per_area', 0.0):.6f}")
        self.metric_labels['coverage_fraction'].config(text=f"{results['coverage_fraction']:.4f}")
        
        # Theory comparison
        expected_good = results.get('expected_good', 0.0)
        self.metric_labels['expected_good'].config(text=f"{expected_good:.1f}")
        self.metric_labels['actual_good'].config(text=str(results.get('isolated_objects', 0)))
        
        # Update plots
        self._update_plots(theory, results, params)
    
    def _update_plots(self, theory, results: Dict[str, Any], params: Dict[str, Any]) -> None:
        """Update plots with analysis results."""
        from ..plots.plot_updater import PlotUpdater
        
        # Clear plots
        self.ax_survival_lambda.clear()
        self.ax_isolated_count.clear()
        
        policy = results.get('policy', 'overlap')
        λ_current = params.get('lambda', 0.001)
        
        # Update survival vs lambda plot
        PlotUpdater.update_survival_vs_lambda(
            self.ax_survival_lambda,
            theory,
            results,
            λ_current,
            policy,
            params
        )
        
        # Update isolated count plot
        H, W = self.stats_view.state.data.geometric_scene.field_shape
        area_px2 = H * W
        
        PlotUpdater.update_isolated_count(
            self.ax_isolated_count,
            theory,
            results,
            area_px2,
            policy,
            params
        )
        
        # Redraw
        self.canvas.draw()

