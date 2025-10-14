"""Metrics panel for displaying crowding analysis results."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, Callable, Optional

from ..controls.parameter_widgets import create_scrollable_frame


class MetricsPanel(ttk.Frame):
    """Panel displaying crowding metrics with dynamic visibility based on policy."""
    
    def __init__(self, parent: tk.Widget):
        """Initialize metrics panel.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Callback placeholders
        self._view_discarded_callback: Optional[Callable[[], None]] = None
        self._view_good_callback: Optional[Callable[[], None]] = None
        
        self._build_ui()
    
    def _build_ui(self) -> None:
        """Build the metrics panel UI."""
        # Create scrollable container
        canvas, scrollable_frame = create_scrollable_frame(self)
        
        # Common metrics (always shown)
        common_metrics = [
            ('Total Objects', 'total_objects'),
            ('', ''),  # Separator
        ]
        
        # Policy-specific metrics (will be shown/hidden)
        self.neighbor_metrics = [
            ('Discarded Objects', 'discarded_by_object'),
            ('', ''),  # Separator
        ]
        
        self.box_metrics = [
            ('Crowded Boxes', 'crowded_boxes'),
            ('Discarded Objects', 'discarded_by_box'),
            ('', ''),  # Separator
        ]
        
        # Shared result metrics
        result_metrics = [
            ('Isolated Objects', 'isolated_objects'),
            ('Isolated/Area (per px²)', 'good_per_area_strict'),
            ('', ''),  # Separator
            ('Coverage Fraction', 'coverage_fraction'),
        ]
        
        self.metric_labels = {}
        row = 0
        
        # Add common metrics
        for name, key in common_metrics:
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
        
        # Neighbor policy specific metrics (with frame for easy show/hide)
        self.neighbor_metrics_frame = ttk.Frame(scrollable_frame)
        self.neighbor_metrics_frame.grid(row=row, column=0, columnspan=2, sticky='ew')
        
        metric_row = 0
        for name, key in self.neighbor_metrics:
            if not name:
                ttk.Separator(self.neighbor_metrics_frame, orient='horizontal').grid(
                    row=metric_row, column=0, columnspan=2, sticky='ew', pady=4
                )
                metric_row += 1
                continue
            
            ttk.Label(
                self.neighbor_metrics_frame,
                text=name + ':',
                font=('TkDefaultFont', 10)
            ).grid(row=metric_row, column=0, sticky='w', padx=4, pady=2)
            
            val_label = ttk.Label(
                self.neighbor_metrics_frame,
                text='—',
                font=('TkDefaultFont', 12, 'bold')
            )
            val_label.grid(row=metric_row, column=1, sticky='e', padx=4, pady=2)
            
            if key:
                self.metric_labels[key] = val_label
            metric_row += 1
        
        row += 1
        
        # Box policy specific metrics (with frame for easy show/hide)
        self.box_metrics_frame = ttk.Frame(scrollable_frame)
        self.box_metrics_frame.grid(row=row, column=0, columnspan=2, sticky='ew')
        
        metric_row = 0
        for name, key in self.box_metrics:
            if not name:
                ttk.Separator(self.box_metrics_frame, orient='horizontal').grid(
                    row=metric_row, column=0, columnspan=2, sticky='ew', pady=4
                )
                metric_row += 1
                continue
            
            ttk.Label(
                self.box_metrics_frame,
                text=name + ':',
                font=('TkDefaultFont', 10)
            ).grid(row=metric_row, column=0, sticky='w', padx=4, pady=2)
            
            val_label = ttk.Label(
                self.box_metrics_frame,
                text='—',
                font=('TkDefaultFont', 12, 'bold')
            )
            val_label.grid(row=metric_row, column=1, sticky='e', padx=4, pady=2)
            
            if key:
                self.metric_labels[key] = val_label
            metric_row += 1
        
        row += 1
        
        # Add result metrics
        for name, key in result_metrics:
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
        
        # Inspector buttons
        inspector_frame = ttk.LabelFrame(scrollable_frame, text='Inspect Objects')
        inspector_frame.grid(row=row, column=0, columnspan=2, sticky='ew', padx=4, pady=8)
        
        ttk.Button(
            inspector_frame,
            text='View Overlapping Objects',
            command=self._on_view_discarded
        ).pack(fill=tk.X, padx=4, pady=2)
        
        ttk.Button(
            inspector_frame,
            text='View Isolated Objects',
            command=self._on_view_good
        ).pack(fill=tk.X, padx=4, pady=2)
    
    def update_policy_visibility(self, policy: str) -> None:
        """Show/hide metric sections based on selected policy.
        
        Args:
            policy: Active policy ('overlap' or 'box')
        """
        if policy == 'overlap':
            self.neighbor_metrics_frame.grid()
            self.box_metrics_frame.grid_remove()
        else:  # box
            self.box_metrics_frame.grid()
            self.neighbor_metrics_frame.grid_remove()
    
    def update_metrics(self, results: Dict[str, Any], expected_good: float) -> None:
        """Update all metric labels with analysis results.
        
        Args:
            results: Analysis results dictionary
            expected_good: Expected number of isolated objects (theory)
        """
        # Common metrics
        self.metric_labels['total_objects'].config(text=str(results['total_objects']))
        
        # Policy-specific metrics
        policy = results.get('policy', 'overlap')
        if policy == 'box':
            self.metric_labels['crowded_boxes'].config(text=str(results.get('crowded_boxes', 0)))
            self.metric_labels['discarded_by_box'].config(text=str(results.get('discarded_by_box', 0)))
        else:  # overlap
            self.metric_labels['discarded_by_object'].config(text=str(results.get('discarded_by_object', 0)))
        
        # Shared result metrics
        self.metric_labels['isolated_objects'].config(text=str(results.get('isolated_objects', 0)))
        self.metric_labels['good_per_area_strict'].config(text=f"{results.get('good_per_area', 0.0):.6f}")
        self.metric_labels['coverage_fraction'].config(text=f"{results['coverage_fraction']:.4f}")
        
        # Theory comparison
        self.metric_labels['expected_good'].config(text=f"{expected_good:.1f}")
        self.metric_labels['actual_good'].config(text=str(results.get('isolated_objects', 0)))
    
    def set_inspector_callbacks(
        self,
        view_discarded_cb: Callable[[], None],
        view_good_cb: Callable[[], None]
    ) -> None:
        """Set callbacks for inspector buttons.
        
        Args:
            view_discarded_cb: Callback for viewing discarded objects
            view_good_cb: Callback for viewing isolated objects
        """
        self._view_discarded_callback = view_discarded_cb
        self._view_good_callback = view_good_cb
    
    def _on_view_discarded(self) -> None:
        """Handle view discarded button click."""
        if self._view_discarded_callback:
            self._view_discarded_callback()
    
    def _on_view_good(self) -> None:
        """Handle view isolated button click."""
        if self._view_good_callback:
            self._view_good_callback()

