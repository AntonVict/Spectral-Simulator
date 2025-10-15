"""Parametric analysis tab for exploring parameter space."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING, Dict, Any, Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from ..statistics_view import StatisticsView


class ParametricTab(ttk.Frame):
    """Parametric analysis with parameter sweeps and theoretical curves."""
    
    def __init__(self, parent: tk.Widget, stats_view: StatisticsView):
        """Initialize parametric tab.
        
        Args:
            parent: Parent widget (notebook)
            stats_view: Reference to main statistics view for shared state
        """
        super().__init__(parent)
        self.stats_view = stats_view
        
        # Configure grid layout: Controls | Results Table | Plots
        self.columnconfigure(0, weight=1, minsize=250)
        self.columnconfigure(1, weight=1, minsize=200)
        self.columnconfigure(2, weight=2, minsize=500)
        self.rowconfigure(0, weight=1)
        
        # Sweep results cache
        self.sweep_results = None
        
        # Build UI components
        self._build_controls()
        self._build_results_table()
        self._build_plots()
    
    def _build_controls(self) -> None:
        """Build sweep configuration controls (left column)."""
        controls_frame = ttk.LabelFrame(self, text='Sweep Configuration')
        controls_frame.grid(row=0, column=0, sticky='nsew', padx=4, pady=4)
        
        # Variable selector
        ttk.Label(controls_frame, text='Sweep Variable:').pack(anchor='w', padx=4, pady=(8, 2))
        
        self.sweep_var = tk.StringVar(value='radius')
        
        var_options = [
            ('Mean Radius', 'radius'),
            ('Density λ', 'lambda'),
            ('Threshold (m/k₀)', 'threshold'),
        ]
        
        for label, value in var_options:
            ttk.Radiobutton(
                controls_frame,
                text=label,
                variable=self.sweep_var,
                value=value,
                command=self._update_sweep_info
            ).pack(anchor='w', padx=4)
        
        ttk.Separator(controls_frame, orient='horizontal').pack(fill=tk.X, padx=4, pady=8)
        
        # Range controls
        ttk.Label(controls_frame, text='Range Settings:').pack(anchor='w', padx=4, pady=(4, 2))
        
        # Min value
        ttk.Label(controls_frame, text='Minimum:').pack(anchor='w', padx=4)
        self.min_value = tk.DoubleVar(value=1.0)
        ttk.Entry(controls_frame, textvariable=self.min_value, width=15).pack(anchor='w', padx=4, pady=(0, 4))
        
        # Max value
        ttk.Label(controls_frame, text='Maximum:').pack(anchor='w', padx=4)
        self.max_value = tk.DoubleVar(value=3.0)
        ttk.Entry(controls_frame, textvariable=self.max_value, width=15).pack(anchor='w', padx=4, pady=(0, 4))
        
        # Step size
        ttk.Label(controls_frame, text='Step Size:').pack(anchor='w', padx=4)
        self.step_size = tk.DoubleVar(value=0.2)
        ttk.Entry(controls_frame, textvariable=self.step_size, width=15).pack(anchor='w', padx=4, pady=(0, 8))
        
        # Info label
        self.sweep_info_label = ttk.Label(
            controls_frame,
            text='Will compute 11 points',
            font=('TkDefaultFont', 9, 'italic'),
            foreground='#2a7f2a'
        )
        self.sweep_info_label.pack(anchor='w', padx=4, pady=(0, 8))
        
        ttk.Separator(controls_frame, orient='horizontal').pack(fill=tk.X, padx=4, pady=8)
        
        # Compute button
        ttk.Button(
            controls_frame,
            text='Compute Curves',
            command=self._compute_sweep
        ).pack(fill=tk.X, padx=4, pady=4)
        
        # Info section
        info_frame = ttk.LabelFrame(controls_frame, text='About')
        info_frame.pack(fill=tk.X, padx=4, pady=8)
        
        ttk.Label(
            info_frame,
            text='Computes theoretical\ncurves by evaluating\nformulas at different\nparameter values.',
            font=('TkDefaultFont', 8),
            justify=tk.LEFT,
            wraplength=200
        ).pack(padx=4, pady=4)
        
        # Initialize info
        self._update_sweep_info()
    
    def _build_results_table(self) -> None:
        """Build results table (middle column)."""
        table_frame = ttk.LabelFrame(self, text='Sweep Results')
        table_frame.grid(row=0, column=1, sticky='nsew', padx=4, pady=4)
        
        # Create scrollable container
        from ..controls.parameter_widgets import create_scrollable_frame
        canvas, self.table_scrollable = create_scrollable_frame(table_frame)
        
        # Placeholder
        self.results_display = ttk.Label(
            self.table_scrollable,
            text='No results yet.\nClick "Compute Curves".',
            font=('TkDefaultFont', 9, 'italic'),
            foreground='#888'
        )
        self.results_display.pack(padx=4, pady=20)
    
    def _build_plots(self) -> None:
        """Build sweep plots (right column)."""
        plots_frame = ttk.LabelFrame(self, text='Parametric Curves')
        plots_frame.grid(row=0, column=2, sticky='nsew', padx=4, pady=4)
        
        # Create matplotlib figure with 2 subplots
        self.figure = Figure(figsize=(7, 8), dpi=100)
        self.figure.subplots_adjust(hspace=0.3, left=0.12, right=0.95, top=0.96, bottom=0.08)
        
        # Plot 1: Survival probability vs parameter
        self.ax_survival = self.figure.add_subplot(211)
        self.ax_survival.set_title('Survival Probability vs Parameter')
        self.ax_survival.set_xlabel('Parameter Value')
        self.ax_survival.set_ylabel('P(survive)')
        self.ax_survival.grid(True, alpha=0.3)
        
        # Plot 2: Expected isolated objects vs parameter
        self.ax_isolated = self.figure.add_subplot(212)
        self.ax_isolated.set_title('Expected Isolated Objects vs Parameter')
        self.ax_isolated.set_xlabel('Parameter Value')
        self.ax_isolated.set_ylabel('Expected Isolated Objects')
        self.ax_isolated.grid(True, alpha=0.3)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=plots_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
    
    def _update_sweep_info(self) -> None:
        """Update the sweep info label."""
        try:
            min_val = self.min_value.get()
            max_val = self.max_value.get()
            step = self.step_size.get()
            
            if step <= 0 or max_val <= min_val:
                self.sweep_info_label.config(text='Invalid range', foreground='#cc0000')
                return
            
            n_points = int((max_val - min_val) / step) + 1
            self.sweep_info_label.config(
                text=f'Will compute {n_points} points',
                foreground='#2a7f2a'
            )
        except:
            self.sweep_info_label.config(text='Invalid values', foreground='#cc0000')
    
    def _compute_sweep(self) -> None:
        """Compute parametric sweep."""
        try:
            min_val = self.min_value.get()
            max_val = self.max_value.get()
            step = self.step_size.get()
            var_type = self.sweep_var.get()
            
            # Validate
            if step <= 0 or max_val <= min_val:
                self.stats_view.log('Error: Invalid sweep range')
                return
            
            # Get current parameters from overview tab
            params = self.stats_view.overview_tab.controls.get_parameters()
            
            # Generate sweep values
            sweep_values = np.arange(min_val, max_val + step/2, step)
            
            # Compute results
            results = self._run_sweep(var_type, sweep_values, params)
            
            # Store results
            self.sweep_results = {
                'variable': var_type,
                'values': sweep_values,
                'results': results
            }
            
            # Update display
            self._update_results_display(var_type, sweep_values, results)
            self._update_plots(var_type, sweep_values, results, params)
            
            self.stats_view.log(f'Parametric sweep completed: {len(sweep_values)} points')
            
        except Exception as e:
            self.stats_view.log(f'Error computing sweep: {str(e)}')
            import traceback
            traceback.print_exc()
    
    def _run_sweep(self, var_type: str, sweep_values: np.ndarray, params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Run the parametric sweep.
        
        Args:
            var_type: Type of variable to sweep
            sweep_values: Array of parameter values
            params: Current analysis parameters
            
        Returns:
            Dictionary with arrays of survival_prob and isolated_count
        """
        from spectral_playground.core.statistics import RadiusDistribution, BooleanModelTheory
        
        survival_probs = []
        isolated_counts = []
        
        # Get current values
        λ = params['intensity']
        mean_r = params['radius_mean']
        std_r = params['radius_std']
        min_r = params['radius_min']
        max_r = params['radius_max']
        m = params['neighbor_threshold']
        k0 = params['box_threshold']
        a = params['box_size']
        mode = params['count_mode']
        policy = params['active_policy']
        
        # Get total objects from current data
        if self.stats_view.state.data.has_geometric_data:
            n_total = len(self.stats_view.state.data.geometric_scene.objects)
        else:
            n_total = 1000  # Default
        
        for value in sweep_values:
            # Modify parameter based on sweep type
            if var_type == 'radius':
                # Vary mean radius, keep std proportional
                mean_r_temp = value
                std_r_temp = std_r * (value / mean_r)
                min_r_temp = max(1.0, min_r * (value / mean_r))
                max_r_temp = max_r * (value / mean_r)
                λ_temp = λ
                m_temp = m
                k0_temp = k0
            elif var_type == 'lambda':
                # Vary intensity
                mean_r_temp = mean_r
                std_r_temp = std_r
                min_r_temp = min_r
                max_r_temp = max_r
                λ_temp = value
                m_temp = m
                k0_temp = k0
            elif var_type == 'threshold':
                # Vary threshold
                mean_r_temp = mean_r
                std_r_temp = std_r
                min_r_temp = min_r
                max_r_temp = max_r
                λ_temp = λ
                m_temp = int(value)
                k0_temp = int(value)
            else:
                raise ValueError(f"Unknown sweep variable: {var_type}")
            
            # Create theory instance
            R_dist = RadiusDistribution(
                mean=mean_r_temp,
                std=std_r_temp,
                r_min=min_r_temp,
                r_max=max_r_temp
            )
            theory = BooleanModelTheory(λ_temp, R_dist)
            
            # Compute survival probability and isolated count
            if policy == 'overlap':
                p_survive = theory.palm_survival_probability(m_temp)
                expected_isolated = n_total * p_survive
            else:  # box policy
                p_crowded = theory.box_crowding_prob(a, k0_temp, mode)
                p_survive = 1 - p_crowded
                expected_isolated = n_total * p_survive
            
            survival_probs.append(p_survive)
            isolated_counts.append(expected_isolated)
        
        return {
            'survival_prob': np.array(survival_probs),
            'isolated_count': np.array(isolated_counts)
        }
    
    def _update_results_display(self, var_type: str, values: np.ndarray, results: Dict[str, np.ndarray]) -> None:
        """Update the results table display."""
        # Clear existing display
        self.results_display.destroy()
        
        # Create new display with table
        table_container = ttk.Frame(self.table_scrollable)
        table_container.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        # Variable name
        var_names = {
            'radius': 'Mean R',
            'lambda': 'λ',
            'threshold': 'Thresh'
        }
        var_name = var_names.get(var_type, 'Value')
        
        # Header
        header_frame = ttk.Frame(table_container)
        header_frame.pack(fill=tk.X)
        
        ttk.Label(
            header_frame,
            text=var_name,
            font=('TkDefaultFont', 8, 'bold'),
            width=8
        ).grid(row=0, column=0, padx=2)
        
        ttk.Label(
            header_frame,
            text='P(surv)',
            font=('TkDefaultFont', 8, 'bold'),
            width=8
        ).grid(row=0, column=1, padx=2)
        
        ttk.Label(
            header_frame,
            text='Isolated',
            font=('TkDefaultFont', 8, 'bold'),
            width=8
        ).grid(row=0, column=2, padx=2)
        
        ttk.Separator(table_container, orient='horizontal').pack(fill=tk.X, pady=2)
        
        # Data rows (show all)
        data_frame = ttk.Frame(table_container)
        data_frame.pack(fill=tk.BOTH, expand=True)
        
        for i, (val, p_surv, isolated) in enumerate(zip(
            values,
            results['survival_prob'],
            results['isolated_count']
        )):
            row_frame = ttk.Frame(data_frame)
            row_frame.pack(fill=tk.X, pady=1)
            
            ttk.Label(
                row_frame,
                text=f'{val:.2f}',
                width=8,
                font=('TkDefaultFont', 8)
            ).grid(row=0, column=0, padx=2)
            
            ttk.Label(
                row_frame,
                text=f'{p_surv:.4f}',
                width=8,
                font=('TkDefaultFont', 8)
            ).grid(row=0, column=1, padx=2)
            
            ttk.Label(
                row_frame,
                text=f'{isolated:.1f}',
                width=8,
                font=('TkDefaultFont', 8)
            ).grid(row=0, column=2, padx=2)
        
        self.results_display = table_container
    
    def _update_plots(self, var_type: str, values: np.ndarray, results: Dict[str, np.ndarray], params: Dict[str, Any]) -> None:
        """Update sweep plots."""
        # Clear plots
        self.ax_survival.clear()
        self.ax_isolated.clear()
        
        # Variable name for labels
        var_labels = {
            'radius': 'Mean Radius (pixels)',
            'lambda': 'Density λ (objects per px²)',
            'threshold': 'Threshold (m or k₀)'
        }
        xlabel = var_labels.get(var_type, 'Parameter Value')
        
        # Get current value to mark on plot
        current_value = None
        if var_type == 'radius':
            current_value = params['radius_mean']
        elif var_type == 'lambda':
            current_value = params['intensity']
        elif var_type == 'threshold':
            policy = params['active_policy']
            current_value = params['neighbor_threshold'] if policy == 'overlap' else params['box_threshold']
        
        # Plot 1: Survival probability
        self.ax_survival.plot(
            values,
            results['survival_prob'],
            'b-',
            linewidth=2,
            label='Theoretical'
        )
        
        if current_value is not None:
            # Find closest value in sweep
            idx = np.argmin(np.abs(values - current_value))
            self.ax_survival.plot(
                values[idx],
                results['survival_prob'][idx],
                'ro',
                markersize=10,
                label='Current'
            )
        
        self.ax_survival.set_xlabel(xlabel, fontsize=10)
        self.ax_survival.set_ylabel('P(survive)', fontsize=10)
        self.ax_survival.set_title('Survival Probability vs Parameter', fontsize=11)
        self.ax_survival.legend(fontsize=9)
        self.ax_survival.grid(True, alpha=0.3)
        
        # Plot 2: Expected isolated objects
        self.ax_isolated.plot(
            values,
            results['isolated_count'],
            'g-',
            linewidth=2,
            label='Expected'
        )
        
        if current_value is not None:
            idx = np.argmin(np.abs(values - current_value))
            self.ax_isolated.plot(
                values[idx],
                results['isolated_count'][idx],
                'ro',
                markersize=10,
                label='Current'
            )
        
        self.ax_isolated.set_xlabel(xlabel, fontsize=10)
        self.ax_isolated.set_ylabel('Expected Isolated Objects', fontsize=10)
        self.ax_isolated.set_title('Expected Isolated Objects vs Parameter', fontsize=11)
        self.ax_isolated.legend(fontsize=9)
        self.ax_isolated.grid(True, alpha=0.3)
        
        # Redraw
        self.canvas.draw()


