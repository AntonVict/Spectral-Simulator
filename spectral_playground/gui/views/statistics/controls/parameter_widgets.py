"""Reusable parameter input widgets for statistics controls."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Tuple


def create_scrollable_frame(parent: tk.Widget) -> Tuple[tk.Canvas, ttk.Frame]:
    """Create a scrollable frame pattern.
    
    Args:
        parent: Parent widget
        
    Returns:
        Tuple of (canvas, scrollable_frame)
    """
    canvas = tk.Canvas(parent, highlightthickness=0)
    scrollbar = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    scrollable_frame.bind(
        '<Configure>',
        lambda e: canvas.configure(scrollregion=canvas.bbox('all'))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
    canvas.configure(yscrollcommand=scrollbar.set)
    
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    return canvas, scrollable_frame


def create_labeled_entry(
    parent: tk.Widget,
    label: str,
    variable: tk.Variable,
    width: int = 12,
    **kwargs
) -> Tuple[ttk.Label, ttk.Entry]:
    """Create a labeled entry widget.
    
    Args:
        parent: Parent widget
        label: Label text
        variable: Tk variable to bind
        width: Entry width
        **kwargs: Additional arguments for label/entry
        
    Returns:
        Tuple of (label_widget, entry_widget)
    """
    label_widget = ttk.Label(parent, text=label)
    entry_widget = ttk.Entry(parent, textvariable=variable, width=width)
    
    return label_widget, entry_widget


def create_labeled_spinbox(
    parent: tk.Widget,
    label: str,
    variable: tk.IntVar,
    from_: int = 1,
    to: int = 20,
    width: int = 10,
    **kwargs
) -> Tuple[ttk.Label, ttk.Spinbox]:
    """Create a labeled spinbox widget.
    
    Args:
        parent: Parent widget
        label: Label text
        variable: Tk IntVar to bind
        from_: Minimum value
        to: Maximum value
        width: Spinbox width
        **kwargs: Additional arguments for spinbox
        
    Returns:
        Tuple of (label_widget, spinbox_widget)
    """
    label_widget = ttk.Label(parent, text=label)
    spinbox_widget = ttk.Spinbox(
        parent,
        from_=from_,
        to=to,
        textvariable=variable,
        width=width,
        **kwargs
    )
    
    return label_widget, spinbox_widget


def create_box_size_slider(
    parent: tk.Widget,
    variable: tk.IntVar,
    update_callback
) -> Tuple[ttk.Scale, ttk.Frame, ttk.Label]:
    """Create box size slider with area display.
    
    Args:
        parent: Parent widget
        variable: IntVar for box size
        update_callback: Callback to update area display
        
    Returns:
        Tuple of (scale, value_frame, area_label)
    """
    # Create scale
    scale = ttk.Scale(
        parent,
        from_=2,
        to=32,
        variable=variable,
        orient=tk.HORIZONTAL,
        command=update_callback
    )
    
    # Create value display frame
    value_frame = ttk.Frame(parent)
    ttk.Label(value_frame, text='a = ').pack(side=tk.LEFT)
    ttk.Label(value_frame, textvariable=variable).pack(side=tk.LEFT)
    ttk.Label(value_frame, text=' → Box area: ').pack(side=tk.LEFT)
    
    # Initial area calculation
    initial_area = variable.get() ** 2
    area_label = ttk.Label(value_frame, text=f'{initial_area} px²', foreground='#2a7f2a')
    area_label.pack(side=tk.LEFT)
    
    return scale, value_frame, area_label

