"""Overlap statistics tab with distribution analysis."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING, Dict, Any, Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from ..statistics_view import StatisticsView


class OverlapTab(ttk.Frame):
    """Neighbor count analysis with distribution and spatial patterns."""
    
    def __init__(self, parent: tk.Widget, stats_view: StatisticsView):
        """Initialize overlap tab.
        
        Args:
            parent: Parent widget (notebook)
            stats_view: Reference to main statistics view for shared state
        """
        super().__init__(parent)
        self.stats_view = stats_view
        
        # Initialize colorbar reference to prevent duplication
        self._heatmap_colorbar = None
        
        # Configure grid layout: Info Panel | Plots
        self.columnconfigure(0, weight=1, minsize=300)
        self.columnconfigure(1, weight=2, minsize=600)
        self.rowconfigure(0, weight=1)
        
        # Build UI components
        self._build_info_panel()
        self._build_plots()
    
    def _build_info_panel(self) -> None:
        """Build info and metrics panel (left column)."""
        info_frame = ttk.LabelFrame(self, text='Overlap Information')
        info_frame.grid(row=0, column=0, sticky='nsew', padx=4, pady=4)
        
        # Create scrollable container
        from ..controls.parameter_widgets import create_scrollable_frame
        canvas, scrollable_frame = create_scrollable_frame(info_frame)
        
        # Policy display
        policy_frame = ttk.LabelFrame(scrollable_frame, text='Active Policy')
        policy_frame.pack(fill=tk.X, padx=4, pady=4)
        
        self.policy_label = ttk.Label(
            policy_frame,
            text='—',
            font=('TkDefaultFont', 10, 'bold'),
            foreground='#2a7f2a'
        )
        self.policy_label.pack(padx=4, pady=4)
        
        # Higher-order overlap table
        table_frame = ttk.LabelFrame(scrollable_frame, text='Neighbor Distribution')
        table_frame.pack(fill=tk.X, padx=4, pady=8)
        
        # Create table header
        header_frame = ttk.Frame(table_frame)
        header_frame.pack(fill=tk.X, padx=4, pady=2)
        
        ttk.Label(
            header_frame,
            text='Neighbors',
            font=('TkDefaultFont', 9, 'bold'),
            width=10
        ).grid(row=0, column=0, padx=2)
        
        ttk.Label(
            header_frame,
            text='Count',
            font=('TkDefaultFont', 9, 'bold'),
            width=8
        ).grid(row=0, column=1, padx=2)
        
        ttk.Label(
            header_frame,
            text='Percent',
            font=('TkDefaultFont', 9, 'bold'),
            width=8
        ).grid(row=0, column=2, padx=2)
        
        # Separator
        ttk.Separator(table_frame, orient='horizontal').pack(fill=tk.X, padx=4, pady=2)
        
        # Table rows (will be populated dynamically)
        self.table_frame = ttk.Frame(table_frame)
        self.table_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)
        
        # Summary statistics
        summary_frame = ttk.LabelFrame(scrollable_frame, text='Summary Statistics')
        summary_frame.pack(fill=tk.X, padx=4, pady=8)
        
        summary_metrics = [
            ('Mean Neighbors', 'mean_neighbors'),
            ('Median Neighbors', 'median_neighbors'),
            ('Max Neighbors', 'max_neighbors'),
        ]
        
        self.summary_labels = {}
        
        for i, (name, key) in enumerate(summary_metrics):
            ttk.Label(
                summary_frame,
                text=name + ':',
                font=('TkDefaultFont', 9)
            ).grid(row=i, column=0, sticky='w', padx=4, pady=2)
            
            val_label = ttk.Label(
                summary_frame,
                text='—',
                font=('TkDefaultFont', 10, 'bold')
            )
            val_label.grid(row=i, column=1, sticky='e', padx=4, pady=2)
            
            self.summary_labels[key] = val_label
        
        # Info section
        info_text_frame = ttk.LabelFrame(scrollable_frame, text='About This View')
        info_text_frame.pack(fill=tk.X, padx=4, pady=8)
        
        ttk.Label(
            info_text_frame,
            text='This tab shows the distribution\n'
                 'of neighbor counts across all\n'
                 'objects, plus a spatial heatmap\n'
                 'revealing where crowding occurs\n'
                 'in the scene.',
            font=('TkDefaultFont', 8),
            justify=tk.LEFT,
            wraplength=250
        ).pack(padx=4, pady=4)
    
    def _build_plots(self) -> None:
        """Build plots panel (right column)."""
        plots_frame = ttk.LabelFrame(self, text='Neighbor Analysis')
        plots_frame.grid(row=0, column=1, sticky='nsew', padx=4, pady=4)
        
        # Create matplotlib figure with 2 subplots
        self.figure = Figure(figsize=(8, 8), dpi=100)
        self.figure.subplots_adjust(hspace=0.3, left=0.10, right=0.95, top=0.96, bottom=0.06)
        
        # Plot 1: Neighbor count histogram
        self.ax_histogram = self.figure.add_subplot(211)
        self.ax_histogram.set_title('Neighbor Count Distribution')
        self.ax_histogram.set_xlabel('Number of Neighbors')
        self.ax_histogram.set_ylabel('Number of Objects')
        self.ax_histogram.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Spatial heatmap showing crowding
        self.ax_heatmap = self.figure.add_subplot(212)
        self.ax_heatmap.set_title('Spatial Crowding Heatmap')
        self.ax_heatmap.set_xlabel('X Position (pixels)')
        self.ax_heatmap.set_ylabel('Y Position (pixels)')
        self.ax_heatmap.set_aspect('equal')
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=plots_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
    
    def update_results(self, results: Dict[str, Any], theory, params: Dict[str, Any]) -> None:
        """Update info and plots with analysis results.
        
        Args:
            results: Analysis results dictionary
            theory: BooleanModelTheory instance
            params: Analysis parameters
        """
        # Update policy display
        policy = results.get('policy', 'overlap')
        policy_name = 'Neighbor Count Policy' if policy == 'overlap' else 'Region Density Policy'
        self.policy_label.config(text=policy_name)
        
        # Get neighbor counts
        neighbor_counts = results.get('neighbor_counts', [])
        
        if len(neighbor_counts) > 0:
            # Compute distribution
            unique_counts, counts = np.unique(neighbor_counts, return_counts=True)
            total = len(neighbor_counts)
            
            # Update table
            self._update_table(unique_counts, counts, total)
            
            # Update summary statistics
            self.summary_labels['mean_neighbors'].config(text=f"{np.mean(neighbor_counts):.2f}")
            self.summary_labels['median_neighbors'].config(text=f"{int(np.median(neighbor_counts))}")
            self.summary_labels['max_neighbors'].config(text=f"{int(np.max(neighbor_counts))}")
            
            # Update plots
            self._update_plots(neighbor_counts, unique_counts, counts, total, params)
        else:
            # Clear if no data
            self._clear_table()
            for label in self.summary_labels.values():
                label.config(text='—')
    
    def _update_table(self, unique_counts, counts, total):
        """Update the neighbor distribution table."""
        # Clear existing rows
        for widget in self.table_frame.winfo_children():
            widget.destroy()
        
        # Show first 10 rows + aggregated rest
        max_display = 10
        
        for i, (k, count) in enumerate(zip(unique_counts, counts)):
            if i >= max_display:
                # Aggregate remaining
                remaining_count = sum(counts[max_display:])
                remaining_pct = remaining_count / total * 100
                
                row_frame = ttk.Frame(self.table_frame)
                row_frame.pack(fill=tk.X, pady=1)
                
                ttk.Label(
                    row_frame,
                    text=f'{int(unique_counts[max_display])}+',
                    width=10,
                    font=('TkDefaultFont', 9)
                ).grid(row=0, column=0, padx=2)
                
                ttk.Label(
                    row_frame,
                    text=str(remaining_count),
                    width=8,
                    font=('TkDefaultFont', 9)
                ).grid(row=0, column=1, padx=2)
                
                ttk.Label(
                    row_frame,
                    text=f'{remaining_pct:.1f}%',
                    width=8,
                    font=('TkDefaultFont', 9)
                ).grid(row=0, column=2, padx=2)
                
                break
            
            percentage = count / total * 100
            
            row_frame = ttk.Frame(self.table_frame)
            row_frame.pack(fill=tk.X, pady=1)
            
            # Highlight isolated objects (k=0)
            if k == 0:
                bg_color = '#e8f5e9'
                font = ('TkDefaultFont', 9, 'bold')
            else:
                bg_color = None
                font = ('TkDefaultFont', 9)
            
            ttk.Label(
                row_frame,
                text=str(int(k)),
                width=10,
                font=font
            ).grid(row=0, column=0, padx=2)
            
            ttk.Label(
                row_frame,
                text=str(count),
                width=8,
                font=font
            ).grid(row=0, column=1, padx=2)
            
            ttk.Label(
                row_frame,
                text=f'{percentage:.1f}%',
                width=8,
                font=font
            ).grid(row=0, column=2, padx=2)
    
    def _clear_table(self):
        """Clear the distribution table."""
        for widget in self.table_frame.winfo_children():
            widget.destroy()
    
    def _update_plots(self, neighbor_counts, unique_counts, counts, total, params):
        """Update histogram and spatial heatmap plots."""
        # Clear plots
        self.ax_histogram.clear()
        self.ax_heatmap.clear()
        
        # Plot 1: Histogram
        colors = ['#50c878' if k == 0 else '#4a90e2' for k in unique_counts]
        
        self.ax_histogram.bar(
            unique_counts,
            counts,
            color=colors,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        
        self.ax_histogram.set_xlabel('Number of Neighbors', fontsize=10)
        self.ax_histogram.set_ylabel('Number of Objects', fontsize=10)
        self.ax_histogram.set_title('Neighbor Count Distribution', fontsize=11)
        self.ax_histogram.grid(True, alpha=0.3, axis='y')
        
        # Add threshold line if applicable
        policy = params.get('active_policy', 'overlap')
        if policy == 'overlap':
            m = params.get('neighbor_threshold', 2)
            self.ax_histogram.axvline(
                m - 0.5,
                color='red',
                linestyle='--',
                linewidth=2,
                label=f'Threshold (m={m})'
            )
            self.ax_histogram.legend(fontsize=9)
        
        # Plot 2: Spatial Heatmap showing where crowding occurs
        self._update_spatial_heatmap(neighbor_counts, params)
        
        # Redraw
        self.canvas.draw()
    
    def _update_spatial_heatmap(self, neighbor_counts: np.ndarray, params: Dict[str, Any]) -> None:
        """Create spatial heatmap showing neighbor count distribution across the scene.
        
        Args:
            neighbor_counts: Array of neighbor counts for each object
            params: Analysis parameters
        """
        # Get geometric scene
        if not self.stats_view.state.data.has_geometric_data:
            self.ax_heatmap.text(0.5, 0.5, 'No geometric data', 
                                ha='center', va='center', fontsize=12)
            return
        
        scene = self.stats_view.state.data.geometric_scene
        objects = scene.objects
        
        if len(objects) == 0:
            self.ax_heatmap.text(0.5, 0.5, 'No objects', 
                                ha='center', va='center', fontsize=12)
            return
        
        # Extract positions and map to neighbor counts
        positions = np.array([obj.position for obj in objects])
        y_coords = positions[:, 0]  # Row (Y)
        x_coords = positions[:, 1]  # Column (X)
        
        # Get scene dimensions
        H, W = scene.field_shape
        
        # Create 2D histogram / heatmap with appropriate binning
        # Use adaptive binning based on scene size
        bins_x = min(50, W // 10)
        bins_y = min(50, H // 10)
        
        # Create weighted 2D histogram where weight = neighbor count
        heatmap, xedges, yedges = np.histogram2d(
            x_coords, y_coords,
            bins=[bins_x, bins_y],
            range=[[0, W], [0, H]],
            weights=neighbor_counts
        )
        
        # Also get count of objects per bin to normalize
        count_map, _, _ = np.histogram2d(
            x_coords, y_coords,
            bins=[bins_x, bins_y],
            range=[[0, W], [0, H]]
        )
        
        # Normalize: average neighbor count per bin (avoid divide by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_heatmap = np.where(count_map > 0, heatmap / count_map, 0)
        
        # Determine color scale based on data range
        # Use max of mean + 2*std to avoid outlier saturation, but ensure minimum range
        mean_neighbors = np.mean(neighbor_counts)
        std_neighbors = np.std(neighbor_counts)
        max_neighbors = np.max(neighbor_counts)
        
        # Set vmax to capture most of the range without saturation
        vmax = min(max_neighbors, max(mean_neighbors + 2 * std_neighbors, 5))
        
        # Plot heatmap
        im = self.ax_heatmap.imshow(
            avg_heatmap.T,
            origin='lower',
            extent=[0, W, 0, H],
            cmap='YlOrRd',
            aspect='equal',
            interpolation='bilinear',
            vmin=0,
            vmax=vmax
        )
        
        # Remove old colorbar if it exists to prevent duplication
        if hasattr(self, '_heatmap_colorbar') and self._heatmap_colorbar is not None:
            try:
                self._heatmap_colorbar.remove()
            except (AttributeError, KeyError, ValueError):
                # Colorbar was already removed or axis state changed
                pass
        
        # Add new colorbar
        self._heatmap_colorbar = self.figure.colorbar(im, ax=self.ax_heatmap, fraction=0.046, pad=0.04)
        self._heatmap_colorbar.set_label('Avg Neighbor Count', fontsize=9)
        
        # Overlay isolated objects as green dots
        isolated_mask = neighbor_counts == 0
        if np.any(isolated_mask):
            self.ax_heatmap.scatter(
                x_coords[isolated_mask],
                y_coords[isolated_mask],
                s=3,
                c='green',
                alpha=0.3,
                marker='.',
                label='Isolated (0 neighbors)'
            )
        
        # Overlay highly crowded objects as red dots
        crowded_threshold = np.percentile(neighbor_counts, 90)
        crowded_mask = neighbor_counts >= crowded_threshold
        if np.any(crowded_mask) and crowded_threshold > 0:
            self.ax_heatmap.scatter(
                x_coords[crowded_mask],
                y_coords[crowded_mask],
                s=5,
                c='red',
                alpha=0.5,
                marker='x',
                label=f'Highly crowded (≥{int(crowded_threshold)} neighbors)'
            )
        
        self.ax_heatmap.set_xlabel('X Position (pixels)', fontsize=10)
        self.ax_heatmap.set_ylabel('Y Position (pixels)', fontsize=10)
        self.ax_heatmap.set_title('Spatial Crowding Heatmap', fontsize=11)
        self.ax_heatmap.legend(fontsize=8, loc='upper right')
        self.ax_heatmap.set_xlim(0, W)
        self.ax_heatmap.set_ylim(0, H)


