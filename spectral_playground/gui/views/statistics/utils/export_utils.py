"""Export utilities for statistics analysis results."""

from __future__ import annotations

from typing import Dict, Any, Callable, Optional
from tkinter import filedialog, messagebox


def export_results_to_csv(
    results: Optional[Dict[str, Any]],
    log_callback: Callable[[str], None]
) -> None:
    """Export analysis results to CSV file.
    
    Args:
        results: Dictionary containing analysis results
        log_callback: Function to log messages
    """
    if results is None:
        messagebox.showwarning('No Results', 'Run analysis first before exporting.')
        return
    
    import pandas as pd
    
    filepath = filedialog.asksaveasfilename(
        defaultextension='.csv',
        filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
    )
    
    if not filepath:
        return
    
    try:
        # Create DataFrame with results
        df = pd.DataFrame([results])
        df.to_csv(filepath, index=False)
        log_callback(f'Results exported to {filepath}')
        messagebox.showinfo('Export Success', f'Results saved to:\n{filepath}')
    except Exception as e:
        messagebox.showerror('Export Error', f'Failed to export:\n{str(e)}')

