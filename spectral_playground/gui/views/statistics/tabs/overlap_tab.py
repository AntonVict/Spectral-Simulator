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
    """Detailed overlap statistics with distributions and higher-order analysis."""
    
    def __init__(self, parent: tk.Widget, stats_view: StatisticsView):
        """Initialize overlap tab.
        
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
                 'objects. Higher-order overlaps\n'
                 '(3+ objects) indicate crowded\n'
                 'regions.',
            font=('TkDefaultFont', 8),
            justify=tk.LEFT,
            wraplength=250
        ).pack(padx=4, pady=4)
    
    def _build_plots(self) -> None:
        """Build plots panel (right column)."""
        plots_frame = ttk.LabelFrame(self, text='Overlap Distributions')
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
        
        # Plot 2: Cumulative distribution
        self.ax_cumulative = self.figure.add_subplot(212)
        self.ax_cumulative.set_title('Cumulative Distribution')
        self.ax_cumulative.set_xlabel('Number of Neighbors')
        self.ax_cumulative.set_ylabel('Cumulative Fraction')
        self.ax_cumulative.grid(True, alpha=0.3)
        
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
        """Update histogram and cumulative plots."""
        # Clear plots
        self.ax_histogram.clear()
        self.ax_cumulative.clear()
        
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
        
        # Plot 2: Cumulative distribution
        cumulative = np.cumsum(counts) / total
        
        self.ax_cumulative.plot(
            unique_counts,
            cumulative,
            'o-',
            color='#4a90e2',
            linewidth=2,
            markersize=6,
            label='Cumulative'
        )
        
        self.ax_cumulative.set_xlabel('Number of Neighbors', fontsize=10)
        self.ax_cumulative.set_ylabel('Cumulative Fraction', fontsize=10)
        self.ax_cumulative.set_title('Cumulative Distribution', fontsize=11)
        self.ax_cumulative.set_ylim(0, 1.05)
        self.ax_cumulative.grid(True, alpha=0.3)
        
        # Add threshold line
        if policy == 'overlap':
            m = params.get('neighbor_threshold', 2)
            self.ax_cumulative.axvline(
                m - 0.5,
                color='red',
                linestyle='--',
                linewidth=2,
                label=f'Threshold (m={m})'
            )
            
            # Add horizontal line showing survival fraction
            if m - 1 < len(unique_counts):
                survival_frac = cumulative[np.where(unique_counts == m - 1)[0][0]] if (m - 1) in unique_counts else 0
                self.ax_cumulative.axhline(
                    survival_frac,
                    color='green',
                    linestyle=':',
                    linewidth=1.5,
                    alpha=0.7,
                    label=f'Survival = {survival_frac:.3f}'
                )
            
            self.ax_cumulative.legend(fontsize=9)
        
        # Redraw
        self.canvas.draw()


