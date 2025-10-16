"""Overlap intensity tab showing how much overlapping objects actually overlap."""

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


class OverlapIntensityTab(ttk.Frame):
    """Overlap intensity analysis - quantifying how severe overlaps are."""
    
    def __init__(self, parent: tk.Widget, stats_view: StatisticsView):
        """Initialize overlap intensity tab.
        
        Args:
            parent: Parent widget (notebook)
            stats_view: Reference to main statistics view for shared state
        """
        super().__init__(parent)
        self.stats_view = stats_view
        
        # Initialize colorbar reference to prevent duplication
        self._scatter_colorbar = None
        
        # Configure grid layout: Info Panel | Plots
        self.columnconfigure(0, weight=1, minsize=300)
        self.columnconfigure(1, weight=2, minsize=600)
        self.rowconfigure(0, weight=1)
        
        # Build UI components
        self._build_info_panel()
        self._build_plots()
    
    def _build_info_panel(self) -> None:
        """Build info and metrics panel (left column)."""
        info_frame = ttk.LabelFrame(self, text='Overlap Intensity Metrics')
        info_frame.grid(row=0, column=0, sticky='nsew', padx=4, pady=4)
        
        # Create scrollable container
        from ..controls.parameter_widgets import create_scrollable_frame
        canvas, scrollable_frame = create_scrollable_frame(info_frame)
        
        # Summary statistics with grid layout (matching overview tab style)
        metrics_frame = ttk.LabelFrame(scrollable_frame, text='Summary Statistics')
        metrics_frame.pack(fill=tk.X, padx=4, pady=4)
        
        # Define key metrics
        key_metrics = [
            ('Total Overlapping Pairs', 'total_pairs'),
            ('Unique Objects Involved', 'unique_objects'),
            ('', ''),  # Separator
            ('Mean Overlap Depth', 'mean_depth'),
            ('Median Overlap Depth', 'median_depth'),
            ('', ''),  # Separator
            ('Mean Coverage Fraction', 'mean_coverage'),
            ('', ''),  # Separator
            ('Pairs with Depth >50%', 'severe_depth'),
            ('Pairs with Coverage = 100%', 'severe_coverage'),
        ]
        
        self.metric_labels = {}
        row = 0
        
        for name, key in key_metrics:
            if not name:
                ttk.Separator(metrics_frame, orient='horizontal').grid(
                    row=row, column=0, columnspan=2, sticky='ew', pady=4, padx=4
                )
                row += 1
                continue
            
            ttk.Label(
                metrics_frame,
                text=name + ':',
                font=('TkDefaultFont', 9)
            ).grid(row=row, column=0, sticky='w', padx=4, pady=2)
            
            val_label = ttk.Label(
                metrics_frame,
                text='—',
                font=('TkDefaultFont', 11, 'bold')
            )
            val_label.grid(row=row, column=1, sticky='e', padx=4, pady=2)
            
            if key:
                self.metric_labels[key] = val_label
            row += 1
        
        # Mathematical formulas (matching coverage tab style)
        formula_frame = ttk.LabelFrame(scrollable_frame, text='Formulas')
        formula_frame.pack(fill=tk.X, padx=4, pady=8)
        
        # Overlap Length
        ttk.Label(
            formula_frame,
            text='overlap_length = (rᵢ + rⱼ) - d',
            font=('TkDefaultFont', 10, 'bold'),
            foreground='#2a7f2a'
        ).pack(padx=4, pady=(4, 2))
        
        ttk.Label(
            formula_frame,
            text='where d = distance between centers',
            font=('TkDefaultFont', 8),
            foreground='#555'
        ).pack(padx=4, pady=(0, 8))
        
        # Overlap Depth
        ttk.Label(
            formula_frame,
            text='Overlap Depth:',
            font=('TkDefaultFont', 9, 'bold')
        ).pack(anchor='w', padx=4, pady=(4, 2))
        
        ttk.Label(
            formula_frame,
            text='depth = overlap_length / (rᵢ + rⱼ)',
            font=('TkDefaultFont', 9),
            foreground='#2a7f2a'
        ).pack(padx=4, pady=(0, 2))
        
        ttk.Label(
            formula_frame,
            text='Range: [0, 1]\n• 0 = just touching\n• 0.5 = halfway (equal sizes)\n• 1 = centers coincide',
            font=('TkDefaultFont', 8),
            foreground='#555',
            justify=tk.LEFT
        ).pack(anchor='w', padx=4, pady=(0, 8))
        
        # Coverage Fraction
        ttk.Label(
            formula_frame,
            text='Coverage Fraction:',
            font=('TkDefaultFont', 9, 'bold')
        ).pack(anchor='w', padx=4, pady=(4, 2))
        
        ttk.Label(
            formula_frame,
            text='coverage = intersection_area / area_smaller',
            font=('TkDefaultFont', 9),
            foreground='#2a7f2a'
        ).pack(padx=4, pady=(0, 2))
        
        ttk.Label(
            formula_frame,
            text='Range: [0, 1]\n• Area-based: actual geometric overlap\n• 1 = smaller circle fully covered',
            font=('TkDefaultFont', 8),
            foreground='#555',
            justify=tk.LEFT
        ).pack(anchor='w', padx=4, pady=(0, 8))
        
        # Key relationship
        ttk.Label(
            formula_frame,
            text='Note:',
            font=('TkDefaultFont', 8, 'bold')
        ).pack(anchor='w', padx=4, pady=(4, 2))
        
        ttk.Label(
            formula_frame,
            text='Non-linear relationship with depth\ndue to circular geometry',
            font=('TkDefaultFont', 8, 'italic'),
            foreground='#2a5f7f'
        ).pack(anchor='w', padx=4, pady=(0, 4))
    
    def _build_plots(self) -> None:
        """Build plots panel (right column)."""
        plots_frame = ttk.Frame(self)
        plots_frame.grid(row=0, column=1, sticky='nsew', padx=4, pady=4)
        plots_frame.rowconfigure(0, weight=1)
        plots_frame.rowconfigure(1, weight=1)
        plots_frame.rowconfigure(2, weight=1)
        plots_frame.columnconfigure(0, weight=1)
        
        # Plot 1: Overlap Depth Histogram
        plot1_frame = ttk.LabelFrame(plots_frame, text='Overlap Depth Distribution')
        plot1_frame.grid(row=0, column=0, sticky='nsew', padx=2, pady=2)
        
        self.fig1 = Figure(figsize=(6, 2.5), dpi=100)
        self.ax1 = self.fig1.add_subplot(111)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=plot1_frame)
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Plot 2: Coverage Fraction Histogram
        plot2_frame = ttk.LabelFrame(plots_frame, text='Coverage Fraction Distribution')
        plot2_frame.grid(row=1, column=0, sticky='nsew', padx=2, pady=2)
        
        self.fig2 = Figure(figsize=(6, 2.5), dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=plot2_frame)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Plot 3: Overlap Depth vs Coverage Fraction Scatter
        plot3_frame = ttk.LabelFrame(plots_frame, text='Overlap Depth vs Coverage Fraction')
        plot3_frame.grid(row=2, column=0, sticky='nsew', padx=2, pady=2)
        
        self.fig3 = Figure(figsize=(6, 2.5), dpi=100)
        self.ax3 = self.fig3.add_subplot(111)
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=plot3_frame)
        self.canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial empty plots
        self._clear_plots()
    
    def _clear_plots(self) -> None:
        """Clear all three plots."""
        self.ax1.clear()
        self.ax1.set_xlabel('Overlap Depth (%)')
        self.ax1.set_ylabel('Number of Pairs')
        self.ax1.set_title('No data')
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.clear()
        self.ax2.set_xlabel('Coverage Fraction (%)')
        self.ax2.set_ylabel('Number of Pairs')
        self.ax2.set_title('No data')
        self.ax2.grid(True, alpha=0.3)
        
        self.ax3.clear()
        self.ax3.set_xlabel('Overlap Depth (%)')
        self.ax3.set_ylabel('Coverage Fraction (%)')
        self.ax3.set_title('No data')
        self.ax3.grid(True, alpha=0.3)
        
        self.canvas1.draw_idle()
        self.canvas2.draw_idle()
        self.canvas3.draw_idle()
    
    def update_results(self, results: Dict[str, Any], theory: Optional[BooleanModelTheory]) -> None:
        """Update tab with new analysis results.
        
        Args:
            results: Analysis results from CrowdingAnalyzer
            theory: Theoretical model (not used yet)
        """
        # Check if we have geometric scene
        if not self.stats_view.state.data.has_geometric_data:
            self._clear_plots()
            self._update_metrics(None)
            return
        
        scene = self.stats_view.state.data.geometric_scene
        
        # Compute overlap intensity metrics
        try:
            overlap_metrics = scene.compute_overlap_metrics()
        except Exception as e:
            self.stats_view.log(f"Error computing overlap metrics: {e}")
            self._clear_plots()
            self._update_metrics(None)
            return
        
        # Update displays
        self._update_metrics(overlap_metrics)
        self._update_plots(overlap_metrics)
    
    def _update_metrics(self, overlap_metrics: Optional[Dict[str, Any]]) -> None:
        """Update metrics labels.
        
        Args:
            overlap_metrics: Results from compute_overlap_metrics() or None
        """
        # Get total number of objects
        scene = self.stats_view.state.data.geometric_scene
        n_total_objects = len(scene.objects) if scene else 0
        
        if overlap_metrics is None or not overlap_metrics['overlap_pairs']:
            self.metric_labels['total_pairs'].config(text='0')
            self.metric_labels['unique_objects'].config(text=f'0 of {n_total_objects} (0%)')
            self.metric_labels['mean_depth'].config(text='—')
            self.metric_labels['median_depth'].config(text='—')
            self.metric_labels['mean_coverage'].config(text='—')
            self.metric_labels['severe_depth'].config(text='—')
            self.metric_labels['severe_coverage'].config(text='—')
            return
        
        pairs = overlap_metrics['overlap_pairs']
        n_pairs = len(pairs)
        
        # Count unique objects involved in overlaps
        unique_objects = set()
        for i, j, _, _, _ in pairs:
            unique_objects.add(i)
            unique_objects.add(j)
        n_unique = len(unique_objects)
        pct_unique = 100 * n_unique / n_total_objects if n_total_objects > 0 else 0
        
        # Extract metrics
        depths = [p[2] for p in pairs]
        coverages = [p[3] for p in pairs]
        
        # Compute statistics
        mean_depth = np.mean(depths)
        median_depth = np.median(depths)
        mean_coverage = np.mean(coverages)
        severe_depth_pct = 100 * np.sum(np.array(depths) > 0.5) / n_pairs
        severe_coverage_pct = 100 * np.sum(np.array(coverages) >= 1.0) / n_pairs
        
        # Update labels (values only, names are in the grid)
        self.metric_labels['total_pairs'].config(text=f'{n_pairs}')
        self.metric_labels['unique_objects'].config(text=f'{n_unique} of {n_total_objects} ({pct_unique:.1f}%)')
        self.metric_labels['mean_depth'].config(text=f'{mean_depth:.1%}')
        self.metric_labels['median_depth'].config(text=f'{median_depth:.1%}')
        self.metric_labels['mean_coverage'].config(text=f'{mean_coverage:.1%}')
        self.metric_labels['severe_depth'].config(text=f'{severe_depth_pct:.1f}%')
        self.metric_labels['severe_coverage'].config(text=f'{severe_coverage_pct:.1f}%')
    
    def _update_plots(self, overlap_metrics: Dict[str, Any]) -> None:
        """Update all three plots.
        
        Args:
            overlap_metrics: Results from compute_overlap_metrics()
        """
        pairs = overlap_metrics['overlap_pairs']
        
        if not pairs:
            self._clear_plots()
            return
        
        # Extract metrics
        depths = np.array([p[2] for p in pairs])
        coverages = np.array([p[3] for p in pairs])
        
        # Plot 1: Overlap Depth Histogram
        self.ax1.clear()
        bins = np.linspace(0, 1, 11)  # 10 bins: 0-10%, 10-20%, ..., 90-100%
        counts, bin_edges, patches = self.ax1.hist(
            depths, bins=bins, edgecolor='black', linewidth=0.5
        )
        
        # Color gradient: green (light) to red (severe)
        for i, patch in enumerate(patches):
            color_val = i / len(patches)
            patch.set_facecolor((color_val, 1 - color_val * 0.7, 0))
        
        self.ax1.set_xlabel('Overlap Depth (%)', fontsize=9)
        self.ax1.set_ylabel('Number of Pairs', fontsize=9)
        self.ax1.set_title(f'Depth Distribution (n={len(depths)})', fontsize=9)
        self.ax1.set_xticks(bins)
        self.ax1.set_xticklabels([f'{int(b*100)}' for b in bins], fontsize=7)
        self.ax1.grid(True, alpha=0.3, axis='y')
        self.fig1.tight_layout()
        
        # Plot 2: Coverage Fraction Histogram
        self.ax2.clear()
        bins_cov = np.linspace(0, 1, 11)
        counts_cov, bin_edges_cov, patches_cov = self.ax2.hist(
            coverages, bins=bins_cov, edgecolor='black', linewidth=0.5
        )
        
        # Color gradient: green (light) to red (severe)
        for i, patch in enumerate(patches_cov):
            color_val = i / len(patches_cov)
            patch.set_facecolor((color_val, 1 - color_val * 0.7, 0))
        
        self.ax2.set_xlabel('Coverage Fraction (%)', fontsize=9)
        self.ax2.set_ylabel('Number of Pairs', fontsize=9)
        self.ax2.set_title(f'Coverage Distribution (n={len(coverages)})', fontsize=9)
        self.ax2.set_xticks(bins_cov)
        self.ax2.set_xticklabels([f'{int(b*100)}' for b in bins_cov], fontsize=7)
        self.ax2.grid(True, alpha=0.3, axis='y')
        self.fig2.tight_layout()
        
        # Plot 3: Overlap Depth vs Coverage Fraction Scatter
        self.ax3.clear()
        
        if len(depths) > 0:
            # Create scatter plot
            # Use alpha for density visualization when many points
            alpha = 0.6 if len(depths) > 100 else 0.8
            markersize = 20 if len(depths) > 500 else 30
            
            scatter = self.ax3.scatter(
                depths * 100, 
                coverages * 100,
                s=markersize,
                alpha=alpha,
                c=depths,  # Color by depth for additional visualization
                cmap='RdYlGn_r',  # Red (severe) to Green (light), reversed
                edgecolors='black',
                linewidth=0.3,
                vmin=0,
                vmax=1
            )
            
            # Remove old colorbar if it exists to prevent duplication
            if hasattr(self, '_scatter_colorbar') and self._scatter_colorbar is not None:
                try:
                    self._scatter_colorbar.remove()
                except (AttributeError, KeyError, ValueError):
                    # Colorbar was already removed or axis state changed
                    pass
            
            # Add new colorbar
            self._scatter_colorbar = self.fig3.colorbar(scatter, ax=self.ax3, pad=0.02)
            self._scatter_colorbar.set_label('Overlap Depth', fontsize=8)
            
            self.ax3.set_xlabel('Overlap Depth (%)', fontsize=9)
            self.ax3.set_ylabel('Coverage Fraction (%)', fontsize=9)
            self.ax3.set_title(f'Overlap Metrics Correlation (n={len(depths)})', fontsize=9)
            self.ax3.set_xlim(-5, 105)
            self.ax3.set_ylim(-5, 105)
            self.ax3.grid(True, alpha=0.3)
            
            # Add text annotation explaining patterns
            self.ax3.text(
                0.02, 0.98, 
                'Coverage based on actual\nintersection area geometry',
                transform=self.ax3.transAxes,
                fontsize=7,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
            )
        else:
            self.ax3.text(0.5, 0.5, 'No overlapping objects', 
                         ha='center', va='center', transform=self.ax3.transAxes)
            self.ax3.set_xlabel('Overlap Depth (%)', fontsize=9)
            self.ax3.set_ylabel('Coverage Fraction (%)', fontsize=9)
        
        self.fig3.tight_layout()
        
        # Redraw canvases
        self.canvas1.draw_idle()
        self.canvas2.draw_idle()
        self.canvas3.draw_idle()

