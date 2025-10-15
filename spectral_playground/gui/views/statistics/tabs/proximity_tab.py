"""Proximity analysis tab showing epsilon-margin isolation counting."""

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
    """Proximity analysis - counting naturally epsilon-isolated objects."""
    
    def __init__(self, parent: tk.Widget, stats_view: StatisticsView):
        """Initialize proximity tab.
        
        Args:
            parent: Parent widget (notebook)
            stats_view: Reference to main statistics view for shared state
        """
        super().__init__(parent)
        self.stats_view = stats_view
        
        # Store latest analysis results
        self.last_margin_analysis = None
        self.last_mean_radius = None
        self.empirical_computed = False
        
        # Configure grid layout: Info Panel | Plots
        self.columnconfigure(0, weight=1, minsize=300)
        self.columnconfigure(1, weight=2, minsize=600)
        self.rowconfigure(0, weight=1)
        
        # Build UI components
        self._build_info_panel()
        self._build_plots()
    
    def _build_info_panel(self) -> None:
        """Build info and metrics panel (left column)."""
        info_frame = ttk.LabelFrame(self, text='Epsilon-Margin Analysis')
        info_frame.grid(row=0, column=0, sticky='nsew', padx=4, pady=4)
        
        # Create scrollable container
        from ..controls.parameter_widgets import create_scrollable_frame
        canvas, scrollable_frame = create_scrollable_frame(info_frame)
        
        # Parameters display
        params_frame = ttk.LabelFrame(scrollable_frame, text='Scene Parameters')
        params_frame.pack(fill=tk.X, padx=4, pady=4)
        
        param_metrics = [
            ('Mean Radius (r̄)', 'mean_radius'),
            ('Object Density (λ)', 'density'),
        ]
        
        self.param_labels = {}
        for name, key in param_metrics:
            row_frame = ttk.Frame(params_frame)
            row_frame.pack(fill=tk.X, padx=4, pady=2)
            
            ttk.Label(row_frame, text=name + ':', font=('TkDefaultFont', 9)).pack(side=tk.LEFT)
            val_label = ttk.Label(row_frame, text='—', font=('TkDefaultFont', 9, 'bold'))
            val_label.pack(side=tk.RIGHT)
            self.param_labels[key] = val_label
        
        # Epsilon-margin statistics (shown after empirical computation)
        self.metrics_frame = ttk.LabelFrame(scrollable_frame, text='Epsilon-Margin Statistics')
        self.metric_labels = {}
        self.metrics_built = False  # Flag to track if we need to build
        
        # Formula section (packed early so we can insert metrics_frame before it)
        self.formula_frame = ttk.LabelFrame(scrollable_frame, text='Theory')
        self.formula_frame.pack(fill=tk.X, padx=4, pady=8)
        
        ttk.Label(
            self.formula_frame,
            text='P(ε-isolated) = exp(-λπ(2r̄+ε)²)',
            font=('TkDefaultFont', 9),
            foreground='#2a7f2a'
        ).pack(padx=4, pady=(4, 2))
        
        ttk.Label(
            self.formula_frame,
            text='Object is ε-isolated if all neighbors have gap ≥ ε',
            font=('TkDefaultFont', 8),
            foreground='#555',
            justify=tk.LEFT
        ).pack(anchor='w', padx=4, pady=(0, 4))
        
        # Empirical computation controls
        empirical_frame = ttk.LabelFrame(scrollable_frame, text='Empirical Validation')
        empirical_frame.pack(fill=tk.X, padx=4, pady=8)
        
        # Epsilon range controls
        ttk.Label(
            empirical_frame,
            text='Epsilon Range (× mean radius):',
            font=('TkDefaultFont', 9, 'bold')
        ).pack(anchor='w', padx=4, pady=(4, 4))
        
        range_frame = ttk.Frame(empirical_frame)
        range_frame.pack(fill=tk.X, padx=4, pady=4)
        
        ttk.Label(range_frame, text='Min:').grid(row=0, column=0, sticky='w', pady=2)
        self.epsilon_min_var = tk.DoubleVar(value=0.0)
        ttk.Entry(range_frame, textvariable=self.epsilon_min_var, width=8).grid(row=0, column=1, padx=4, pady=2)
        
        ttk.Label(range_frame, text='Max:').grid(row=0, column=2, sticky='w', padx=(8, 0), pady=2)
        self.epsilon_max_var = tk.DoubleVar(value=2.0)
        ttk.Entry(range_frame, textvariable=self.epsilon_max_var, width=8).grid(row=0, column=3, padx=4, pady=2)
        
        ttk.Label(range_frame, text='Points:').grid(row=1, column=0, sticky='w', pady=2)
        self.epsilon_points_var = tk.IntVar(value=3)
        ttk.Entry(range_frame, textvariable=self.epsilon_points_var, width=8).grid(row=1, column=1, padx=4, pady=2)
        
        # Add traces to variables to update info in real-time
        self.epsilon_min_var.trace_add('write', lambda *args: self._update_empirical_info())
        self.epsilon_max_var.trace_add('write', lambda *args: self._update_empirical_info())
        self.epsilon_points_var.trace_add('write', lambda *args: self._update_empirical_info())
        
        # Info label
        self.empirical_info_label = ttk.Label(
            empirical_frame,
            text='Will compute 3 points: [0 to 2r̄]',
            font=('TkDefaultFont', 8, 'italic'),
            foreground='#2a7f2a'
        )
        self.empirical_info_label.pack(anchor='w', padx=4, pady=(4, 8))
        
        # Compute button
        self.compute_empirical_btn = ttk.Button(
            empirical_frame,
            text='Compute Empirical',
            command=self._compute_empirical
        )
        self.compute_empirical_btn.pack(fill=tk.X, padx=4, pady=4)
        
        # Status label
        self.empirical_status_label = ttk.Label(
            empirical_frame,
            text='Not computed',
            font=('TkDefaultFont', 8, 'italic'),
            foreground='#888'
        )
        self.empirical_status_label.pack(padx=4, pady=(2, 4))
    
    def _build_plots(self) -> None:
        """Build plots panel (right column)."""
        plots_frame = ttk.Frame(self)
        plots_frame.grid(row=0, column=1, sticky='nsew', padx=4, pady=4)
        plots_frame.rowconfigure(0, weight=1)
        plots_frame.rowconfigure(1, weight=1)
        plots_frame.columnconfigure(0, weight=1)
        
        # Plot 1: Objects Kept vs Epsilon (Main plot!)
        plot1_frame = ttk.LabelFrame(plots_frame, text='Objects Kept vs Epsilon-Margin')
        plot1_frame.grid(row=0, column=0, sticky='nsew', padx=2, pady=2)
        
        self.fig1 = Figure(figsize=(6, 3.5), dpi=100)
        self.ax1 = self.fig1.add_subplot(111)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=plot1_frame)
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Plot 2: Survival Probability vs Epsilon
        plot2_frame = ttk.LabelFrame(plots_frame, text='Survival Probability vs Epsilon-Margin')
        plot2_frame.grid(row=1, column=0, sticky='nsew', padx=2, pady=2)
        
        self.fig2 = Figure(figsize=(6, 3.5), dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=plot2_frame)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial empty plots
        self._clear_plots()
    
    def _clear_plots(self) -> None:
        """Clear both plots."""
        self.ax1.clear()
        self.ax1.set_xlabel('Epsilon Margin (pixels)')
        self.ax1.set_ylabel('Number of Objects Kept')
        self.ax1.set_title('No data')
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.clear()
        self.ax2.set_xlabel('Epsilon Margin (pixels)')
        self.ax2.set_ylabel('Survival Probability')
        self.ax2.set_title('No data')
        self.ax2.grid(True, alpha=0.3)
        
        self.canvas1.draw_idle()
        self.canvas2.draw_idle()
    
    def update_results(self, results: Dict[str, Any], theory: Optional[BooleanModelTheory]) -> None:
        """Update tab with new analysis results.
        
        Computes ONLY theoretical curves by default. Empirical can be computed on-demand
        via the "Compute Empirical" button.
        
        Args:
            results: Analysis results from CrowdingAnalyzer
            theory: Theoretical model (not used yet)
        """
        # Check if we have geometric scene
        if not self.stats_view.state.data.has_geometric_data:
            self._clear_plots()
            self._update_metrics(None, None)
            self.empirical_computed = False
            self.empirical_status_label.config(text='')
            return
        
        scene = self.stats_view.state.data.geometric_scene
        
        # Compute mean radius and other parameters
        radii = np.array([obj.radius for obj in scene.objects])
        if len(radii) == 0:
            self._clear_plots()
            self._update_metrics(None, None)
            self.empirical_computed = False
            self.empirical_status_label.config(text='')
            return
        
        mean_radius = np.mean(radii)
        
        # Theoretical: many points for smooth curve (instant!)
        epsilon_theoretical = np.linspace(0, 3 * mean_radius, 50)
        
        # Compute ONLY theoretical analysis (no empirical)
        try:
            margin_analysis = scene.compute_epsilon_margin_analysis(
                [],  # No empirical computation
                epsilon_theoretical
            )
        except Exception as e:
            self.stats_view.log(f"Error computing theoretical analysis: {e}")
            self._clear_plots()
            self._update_metrics(None, None)
            return
        
        # Store for later empirical computation
        self.last_margin_analysis = margin_analysis
        self.last_mean_radius = mean_radius
        self.empirical_computed = False
        self.empirical_status_label.config(text='Not computed', foreground='#888')
        
        # Update displays (theoretical only)
        self._update_metrics(margin_analysis, mean_radius)
        self._update_plots(margin_analysis, mean_radius)
    
    def _update_empirical_info(self) -> None:
        """Update the empirical computation info label."""
        try:
            n_points = self.epsilon_points_var.get()
            eps_min = self.epsilon_min_var.get()
            eps_max = self.epsilon_max_var.get()
            
            if n_points <= 0 or eps_max <= eps_min:
                self.empirical_info_label.config(text='Invalid range', foreground='#cc0000')
            else:
                self.empirical_info_label.config(
                    text=f'Will compute {n_points} points: [{eps_min:.1f}r̄ to {eps_max:.1f}r̄]',
                    foreground='#2a7f2a'
                )
        except:
            self.empirical_info_label.config(text='Invalid values', foreground='#cc0000')
    
    def _compute_empirical(self) -> None:
        """Compute empirical values at specified epsilon points."""
        if not self.stats_view.state.data.has_geometric_data or self.last_mean_radius is None:
            return
        
        scene = self.stats_view.state.data.geometric_scene
        n_objects = len(scene.objects)
        mean_radius = self.last_mean_radius
        
        # Get epsilon range from UI
        try:
            eps_min = self.epsilon_min_var.get()
            eps_max = self.epsilon_max_var.get()
            n_points = self.epsilon_points_var.get()
            
            if n_points <= 0 or eps_max <= eps_min:
                self.stats_view.log('Error: Invalid epsilon range')
                return
            
            # Generate linearly spaced epsilon values in multiples of mean_radius
            epsilon_empirical = list(np.linspace(eps_min * mean_radius, eps_max * mean_radius, n_points))
            eps_desc = f'[{eps_min:.1f}r̄ to {eps_max:.1f}r̄, {n_points} pts]'
        except Exception as e:
            self.stats_view.log(f'Error: Invalid epsilon parameters: {e}')
            return
        
        # Disable button and show status
        self.compute_empirical_btn.config(state='disabled')
        self.empirical_status_label.config(text='Computing...', foreground='#cc7700')
        self.stats_view.log(f'Computing empirical ε-isolated counts for {n_objects} objects at {n_points} epsilon values...')
        self.update()  # Force UI update
        
        try:
            # Compute empirical values
            kept_empirical = []
            for i, eps in enumerate(epsilon_empirical):
                self.empirical_status_label.config(
                    text=f'Computing {i+1}/{n_points}...',
                    foreground='#cc7700'
                )
                self.update()
                
                n_isolated = scene._count_epsilon_isolated(eps)
                kept_empirical.append(n_isolated)
            
            # Add empirical results to last_margin_analysis
            self.last_margin_analysis['epsilon_empirical'] = epsilon_empirical
            self.last_margin_analysis['kept_empirical'] = kept_empirical
            kept_empirical_pct = [100 * k / n_objects for k in kept_empirical] if n_objects > 0 else []
            self.last_margin_analysis['kept_empirical_pct'] = kept_empirical_pct
            
            # Extend theoretical curve if empirical range is larger
            max_emp_epsilon = max(epsilon_empirical)
            max_theory_epsilon = max(self.last_margin_analysis['epsilon_theoretical'])
            
            if max_emp_epsilon > max_theory_epsilon:
                # Recompute theoretical with extended range
                epsilon_theoretical_extended = np.linspace(0, max_emp_epsilon * 1.1, 50)
                extended_analysis = scene.compute_epsilon_margin_analysis(
                    [],  # No empirical
                    epsilon_theoretical_extended
                )
                # Update theoretical data only
                self.last_margin_analysis['epsilon_theoretical'] = extended_analysis['epsilon_theoretical']
                self.last_margin_analysis['kept_theoretical'] = extended_analysis['kept_theoretical']
                self.last_margin_analysis['kept_theoretical_pct'] = extended_analysis['kept_theoretical_pct']
                self.last_margin_analysis['survival_prob_theoretical'] = extended_analysis['survival_prob_theoretical']
            
            self.empirical_computed = True
            self.empirical_status_label.config(text='✓ Computed', foreground='#2a7f2a')
            self.stats_view.log(f'Empirical computation complete: {n_points} points at {eps_desc}')
            
            # Update displays with empirical data
            self._update_metrics(self.last_margin_analysis, self.last_mean_radius)
            self._update_plots(self.last_margin_analysis, self.last_mean_radius)
            
        except Exception as e:
            self.stats_view.log(f'Error computing empirical values: {e}')
            self.empirical_status_label.config(text='Error', foreground='#cc0000')
            import traceback
            traceback.print_exc()
        finally:
            self.compute_empirical_btn.config(state='normal')
    
    def _build_metrics_panel(self, margin_analysis: Dict[str, Any]) -> None:
        """Build metrics panel dynamically showing empirical results.
        
        Args:
            margin_analysis: Results containing empirical epsilon values
        """
        eps_emp = margin_analysis['epsilon_empirical']
        
        # Check if we need to rebuild (different number of epsilon values)
        if self.metrics_built and hasattr(self, 'epsilon_metric_labels'):
            if len(self.epsilon_metric_labels) == len(eps_emp):
                # Same number of points, just update values
                self._update_metrics_values(margin_analysis)
                return
            else:
                # Different number of points, need to rebuild
                self.metrics_built = False
        
        # Clear any existing content first
        for widget in self.metrics_frame.winfo_children():
            widget.destroy()
        
        # Pack the metrics frame before formula_frame (if not already packed)
        if not self.metrics_frame.winfo_ismapped():
            self.metrics_frame.pack(fill=tk.X, padx=4, pady=8, before=self.formula_frame)
        
        # Create grid for metrics
        eps_emp = margin_analysis['epsilon_empirical']
        kept_emp = margin_analysis['kept_empirical']
        n_total = len(self.stats_view.state.data.geometric_scene.objects)
        mean_radius = self.last_mean_radius
        
        # Store labels for later updates
        self.epsilon_metric_labels = []
        
        row = 0
        for i, (eps, kept) in enumerate(zip(eps_emp, kept_emp)):
            # Determine epsilon label
            eps_ratio = eps / mean_radius if mean_radius > 0 else 0
            
            if abs(eps) < 0.01:
                eps_label = 'At ε = 0:'
            elif abs(eps_ratio - 0.5) < 0.1:
                eps_label = 'At ε = 0.5r̄:'
            elif abs(eps_ratio - 1.0) < 0.1:
                eps_label = 'At ε = r̄:'
            elif abs(eps_ratio - 2.0) < 0.1:
                eps_label = 'At ε = 2r̄:'
            elif abs(eps_ratio - 3.0) < 0.1:
                eps_label = 'At ε = 3r̄:'
            else:
                eps_label = f'At ε = {eps:.1f} px:'
            
            # Section header
            header = ttk.Label(
                self.metrics_frame,
                text=eps_label,
                font=('TkDefaultFont', 9, 'bold'),
                foreground='#2a7f2a'
            )
            header.grid(row=row, column=0, columnspan=2, sticky='w', padx=4, pady=(4, 2))
            row += 1
            
            # Isolated count
            ttk.Label(
                self.metrics_frame,
                text='  ε-Isolated:',
                font=('TkDefaultFont', 9)
            ).grid(row=row, column=0, sticky='w', padx=4, pady=2)
            
            kept_pct = 100 * kept / n_total if n_total > 0 else 0
            kept_label = ttk.Label(
                self.metrics_frame,
                text=f'{int(kept)} ({kept_pct:.1f}%)',
                font=('TkDefaultFont', 11, 'bold')
            )
            kept_label.grid(row=row, column=1, sticky='e', padx=4, pady=2)
            row += 1
            
            # Non-isolated count
            discarded = n_total - kept
            ttk.Label(
                self.metrics_frame,
                text='  Non-isolated:',
                font=('TkDefaultFont', 9)
            ).grid(row=row, column=0, sticky='w', padx=4, pady=2)
            
            disc_pct = 100 * discarded / n_total if n_total > 0 else 0
            disc_label = ttk.Label(
                self.metrics_frame,
                text=f'{int(discarded)} ({disc_pct:.1f}%)',
                font=('TkDefaultFont', 11, 'bold')
            )
            disc_label.grid(row=row, column=1, sticky='e', padx=4, pady=2)
            row += 1
            
            # Store labels for updates
            self.epsilon_metric_labels.append({
                'header': header,
                'kept': kept_label,
                'discarded': disc_label,
                'epsilon': eps
            })
            
            # Add separator except after last item
            if i < len(eps_emp) - 1:
                ttk.Separator(self.metrics_frame, orient='horizontal').grid(
                    row=row, column=0, columnspan=2, sticky='ew', pady=4, padx=4
                )
                row += 1
        
        self.metrics_built = True
    
    def _update_metrics_values(self, margin_analysis: Dict[str, Any]) -> None:
        """Update existing metrics panel with new values.
        
        Args:
            margin_analysis: Results containing empirical epsilon values
        """
        if not self.metrics_built or not hasattr(self, 'epsilon_metric_labels'):
            return
        
        eps_emp = margin_analysis['epsilon_empirical']
        kept_emp = margin_analysis['kept_empirical']
        n_total = len(self.stats_view.state.data.geometric_scene.objects)
        
        # Update each metric
        for i, metric_set in enumerate(self.epsilon_metric_labels):
            if i < len(kept_emp):
                kept = kept_emp[i]
                discarded = n_total - kept
                kept_pct = 100 * kept / n_total if n_total > 0 else 0
                disc_pct = 100 * discarded / n_total if n_total > 0 else 0
                
                metric_set['kept'].config(text=f'{int(kept)} ({kept_pct:.1f}%)')
                metric_set['discarded'].config(text=f'{int(discarded)} ({disc_pct:.1f}%)')
    
    def _update_metrics(self, margin_analysis: Optional[Dict[str, Any]], mean_radius: Optional[float]) -> None:
        """Update metrics labels.
        
        Args:
            margin_analysis: Results from compute_epsilon_margin_analysis() or None
            mean_radius: Mean object radius or None
        """
        n_total = len(self.stats_view.state.data.geometric_scene.objects) if self.stats_view.state.data.has_geometric_data else 0
        
        # Update parameters
        if mean_radius is not None:
            self.param_labels['mean_radius'].config(text=f'{mean_radius:.2f} px')
            
            H, W = self.stats_view.state.data.geometric_scene.field_shape
            area = H * W
            density = n_total / area
            self.param_labels['density'].config(text=f'{density:.6f} /px²')
        else:
            self.param_labels['mean_radius'].config(text='—')
            self.param_labels['density'].config(text='—')
        
        # Build/update metrics panel if empirical data exists
        if margin_analysis is not None and len(margin_analysis.get('epsilon_empirical', [])) > 0:
            self._build_metrics_panel(margin_analysis)
    
    def _update_plots(self, margin_analysis: Dict[str, Any], mean_radius: float) -> None:
        """Update both plots.
        
        Args:
            margin_analysis: Results from compute_epsilon_margin_analysis()
            mean_radius: Mean object radius
        """
        n_total = len(self.stats_view.state.data.geometric_scene.objects)
        
        # Extract data
        eps_emp = margin_analysis['epsilon_empirical']
        kept_emp = margin_analysis['kept_empirical']
        eps_theory = margin_analysis['epsilon_theoretical']
        kept_theory = margin_analysis['kept_theoretical']
        survival_prob = margin_analysis['survival_prob_theoretical']
        
        # Determine plot x-axis range based on available data
        all_epsilon = list(eps_theory) + list(eps_emp)
        max_epsilon = max(all_epsilon) if all_epsilon else 3 * mean_radius
        x_limit = max_epsilon * 1.05
        
        # Plot 1: Objects Kept vs Epsilon
        self.ax1.clear()
        
        if len(eps_theory) > 0:
            # Theoretical curve (smooth)
            self.ax1.plot(eps_theory, kept_theory, 'b-', linewidth=2, 
                         label='Theoretical', alpha=0.8)
        
        # Empirical points (validation)
        if len(eps_emp) > 0:
            self.ax1.plot(eps_emp, kept_emp, 'ro', markersize=8, 
                         label='Empirical', zorder=10)
        
        self.ax1.set_xlabel('Epsilon Margin (pixels)', fontsize=10)
        self.ax1.set_ylabel('Number of ε-Isolated Objects', fontsize=10)
        self.ax1.set_title(f'Naturally ε-Isolated Objects (N={n_total})', fontsize=10)
        self.ax1.set_xlim(0, x_limit)
        self.ax1.set_ylim(0, n_total * 1.05)
        self.ax1.legend(fontsize=9)
        self.ax1.grid(True, alpha=0.3)
        
        # Add vertical line at mean radius
        if mean_radius < x_limit:
            self.ax1.axvline(mean_radius, color='gray', linestyle='--', 
                            alpha=0.5, linewidth=1)
            self.ax1.text(mean_radius, n_total * 0.95, f'  r̄={mean_radius:.1f}', 
                         fontsize=8, color='gray')
        
        self.fig1.tight_layout()
        
        # Plot 2: Survival Probability vs Epsilon
        self.ax2.clear()
        
        if len(eps_theory) > 0:
            # Theoretical curve
            self.ax2.plot(eps_theory, survival_prob, 'g-', linewidth=2, 
                         label='Theoretical', alpha=0.8)
        
        # Empirical points (calculated from kept_emp)
        if len(eps_emp) > 0:
            survival_emp = np.array(kept_emp) / n_total if n_total > 0 else []
            self.ax2.plot(eps_emp, survival_emp, 'ro', markersize=8, 
                         label='Empirical', zorder=10)
        
        self.ax2.set_xlabel('Epsilon Margin (pixels)', fontsize=10)
        self.ax2.set_ylabel('Isolation Probability', fontsize=10)
        self.ax2.set_title('P(object is ε-isolated)', fontsize=10)
        self.ax2.set_xlim(0, x_limit)
        self.ax2.set_ylim(0, 1.05)
        self.ax2.legend(fontsize=9)
        self.ax2.grid(True, alpha=0.3)
        
        # Add vertical line at mean radius
        if mean_radius < x_limit:
            self.ax2.axvline(mean_radius, color='gray', linestyle='--', 
                            alpha=0.5, linewidth=1)
        
        self.fig2.tight_layout()
        
        # Redraw canvases
        self.canvas1.draw_idle()
        self.canvas2.draw_idle()
