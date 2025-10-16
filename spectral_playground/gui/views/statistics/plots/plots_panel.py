"""Plots panel for statistics view with matplotlib integration."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class PlotsPanel(ttk.Frame):
    """Panel containing analysis plots with matplotlib integration."""
    
    def __init__(self, parent: tk.Widget):
        """Initialize plots panel.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Create matplotlib figure with 3 subplots
        self.figure = Figure(figsize=(7, 10), dpi=100)
        self.figure.subplots_adjust(hspace=0.35, left=0.13, right=0.95, top=0.96, bottom=0.06)
        
        # Plot 1: Survival probability vs intensity
        self.ax_survival_lambda = self.figure.add_subplot(311)
        self.ax_survival_lambda.set_title('Survival Probability vs Density')
        self.ax_survival_lambda.set_xlabel('Density λ (objects per px²)')
        self.ax_survival_lambda.set_ylabel('P(survive)')
        self.ax_survival_lambda.grid(True, alpha=0.3)
        
        # Plot 2: Survival probability vs threshold
        self.ax_survival_threshold = self.figure.add_subplot(312)
        self.ax_survival_threshold.set_title('Survival Probability vs Threshold')
        self.ax_survival_threshold.set_xlabel('Threshold parameter')
        self.ax_survival_threshold.set_ylabel('P(survive)')
        self.ax_survival_threshold.grid(True, alpha=0.3)
        
        # Plot 3: Isolated objects vs density
        self.ax_isolated_count = self.figure.add_subplot(313)
        self.ax_isolated_count.set_title('Isolated Objects vs Density')
        self.ax_isolated_count.set_xlabel('Total Objects (n)')
        self.ax_isolated_count.set_ylabel('Isolated Objects')
        self.ax_isolated_count.grid(True, alpha=0.3)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
    
    def clear_all(self) -> None:
        """Clear all plot axes."""
        self.ax_survival_lambda.clear()
        self.ax_survival_threshold.clear()
        self.ax_isolated_count.clear()
    
    def draw(self) -> None:
        """Redraw the canvas."""
        self.canvas.draw()

