from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .bottom_panel import BottomPanel
from .data_manager import (
    generate_dataset,
    load_dataset,
    save_composite,
    save_dataset,
    save_plots,
)
from .sidebar import Sidebar
from .state import PlaygroundState
from .viewer import ViewerPanel


class PlaygroundGUI(tk.Tk):
    """Interactive GUI focused on visualising spectral datasets."""

    def __init__(self) -> None:
        super().__init__()
        self.title('Spectral Playground - Visualizer')
        self.geometry('1600x950')  # Increased size for better visibility

        self.state = PlaygroundState()
        self.save_root = self._setup_save_directory()
        self.state.save_directory = self.save_root

        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=3)  # Top area (composite view) gets 75% priority
        self.rowconfigure(1, weight=1)  # Bottom area (analysis panels) gets 25% priority, now resizable!

        self.sidebar = Sidebar(self, self._log)
        self.sidebar.grid(row=0, column=0, sticky='nsw', padx=4, pady=4)

        self.viewer = ViewerPanel(
            self,
            self.state,
            on_generate=self.generate_data,
            on_load=self.load_dataset,
            on_save_dataset=self.save_dataset,
            on_save_composite=self.save_composite,
            on_export_plots=self.export_plots,
            on_open_folder=self.open_save_directory,
            on_channels_changed=self._on_channels_changed,
            on_expand_composite=self._expand_composite_view,
        )
        self.viewer.grid(row=0, column=1, sticky='nsew')

        self.bottom = BottomPanel(
            self, 
            self.state, 
            on_fluor_selection_changed=self._on_fluor_selection_changed,
            on_open_full_inspector=self._open_full_inspector,
            get_data_callback=lambda: self.state.data,
            get_fluorophore_names_callback=self._get_fluorophore_names
        )
        self.bottom.grid(row=1, column=0, columnspan=2, sticky='nsew', padx=8, pady=8)
        self.bottom.configure_expanders(self._expand_spectral_view, self._expand_abundance_view)

        self.after(100, self._show_startup_message)

    # ------------------------------------------------------------------
    # Actions

    def generate_data(self) -> None:
        try:
            # Validate channels before generating
            is_valid, error_msg = self.sidebar.panels['channels'].manager.validate_all()
            if not is_valid:
                messagebox.showwarning('Invalid Channel Configuration', error_msg)
                self._log(f'Channel validation failed: {error_msg}')
                return
            
            cfg = self.sidebar.get_generation_config()
            fluorophores = self.sidebar.get_fluorophores()
            if not fluorophores:
                messagebox.showwarning('No Fluorophores', 'Add at least one fluorophore before generating data.')
                return
            objects = self.sidebar.get_object_specs()
            data = generate_dataset(cfg, fluorophores, objects)
            self.state.data = data
            self._after_dataset_change(f"Generated dataset (L={data.Y.shape[0]}, K={data.A.shape[0]})")
        except Exception as exc:  # pragma: no cover - UI feedback
            messagebox.showerror('Generation failed', str(exc))
            self._log(f'Error generating data: {exc}')

    def load_dataset(self) -> None:
        filepath = filedialog.askopenfilename(
            title='Load Dataset',
            initialdir=self._default_path('datasets'),
            filetypes=[('NumPy archive', '*.npz'), ('All files', '*.*')],
        )
        if not filepath:
            return
        try:
            data = load_dataset(filepath)
            self.state.data = data
            self.sidebar.apply_dataset(data.spectral, data.field)
            self.sidebar.set_fluorophores_from_dataset(data.spectral.fluors)
            rel = os.path.relpath(filepath, self.save_root) if filepath.startswith(self.save_root) else filepath
            self._after_dataset_change(f'Loaded dataset: {rel}')
        except Exception as exc:  # pragma: no cover - UI feedback
            messagebox.showerror('Load failed', str(exc))
            self._log(f'Error loading dataset: {exc}')

    def save_dataset(self) -> None:
        if not self.state.data.has_data:
            messagebox.showinfo('No Data', 'Generate or load data before saving.')
            return
        default_name = 'dataset.npz'
        filepath = filedialog.asksaveasfilename(
            title='Save Dataset',
            initialdir=self._default_path('datasets'),
            initialfile=default_name,
            defaultextension='.npz',
            filetypes=[('NumPy archive', '*.npz'), ('All files', '*.*')],
        )
        if not filepath:
            return
        try:
            save_dataset(filepath, self.state.data)
            rel = os.path.relpath(filepath, self.save_root) if filepath.startswith(self.save_root) else filepath
            self._log(f'Saved dataset: {rel}')
        except Exception as exc:  # pragma: no cover
            messagebox.showerror('Save failed', str(exc))
            self._log(f'Error saving dataset: {exc}')

    def save_composite(self) -> None:
        if not self.state.data.has_data:
            messagebox.showinfo('No Data', 'Generate or load data before saving.')
            return
        rgb = self.viewer.composite_view.latest_rgb
        if rgb is None:
            messagebox.showwarning('Unavailable', 'Render the composite view before saving.')
            return
        filepath = filedialog.asksaveasfilename(
            title='Save Composite Image',
            initialdir=self._default_path('images'),
            initialfile='composite.png',
            defaultextension='.png',
            filetypes=[('PNG image', '*.png'), ('JPEG image', '*.jpg;*.jpeg')],
        )
        if not filepath:
            return
        fmt = 'JPEG' if filepath.lower().endswith(('.jpg', '.jpeg')) else 'PNG'
        try:
            save_composite(filepath, rgb, fmt)
            rel = os.path.relpath(filepath, self.save_root) if filepath.startswith(self.save_root) else filepath
            self._log(f'Saved composite image: {rel}')
        except Exception as exc:  # pragma: no cover
            messagebox.showerror('Save failed', str(exc))
            self._log(f'Error saving composite: {exc}')

    def export_plots(self) -> None:
        if not self.state.data.has_data:
            messagebox.showinfo('No Data', 'Generate or load data before exporting.')
            return
        output_dir = filedialog.askdirectory(
            title='Choose export directory',
            initialdir=self._default_path('exports'),
        )
        if not output_dir:
            return
        rgb = self.viewer.composite_view.latest_rgb
        if rgb is None:
            self._log('Composite view not rendered; rendering before export.')
            self.update_visualisation()
            rgb = self.viewer.composite_view.latest_rgb
        try:
            save_plots(
                output_dir,
                rgb,
                self.bottom.spectral_panel.figure,
                self.bottom.abundance_panel.figure,
                self.state.data.A,
                self.state.data.field.shape if self.state.data.field else None,
            )
            self._log(f'Exported plots to: {output_dir}')
        except Exception as exc:  # pragma: no cover
            messagebox.showerror('Export failed', str(exc))
            self._log(f'Error exporting plots: {exc}')

    def open_save_directory(self) -> None:
        if not self.save_root:
            return
        try:
            if os.name == 'nt':
                os.startfile(self.save_root)  # type: ignore[attr-defined]
            elif sys.platform == 'darwin':
                subprocess.run(['open', self.save_root], check=False)
            else:
                subprocess.run(['xdg-open', self.save_root], check=False)
            self._log(f'Opened save directory: {self.save_root}')
        except Exception as exc:  # pragma: no cover
            self._log(f'Could not open save directory: {exc}')

    # ------------------------------------------------------------------
    # Visual updates

    def _after_dataset_change(self, message: str) -> None:
        channel_names = [ch.name for ch in self.state.data.spectral.channels]
        self.viewer.set_channels(channel_names)
        fluor_names = [fl.name or f'F{i + 1}' for i, fl in enumerate(self.state.data.spectral.fluors)]
        self.bottom.set_fluorophores(fluor_names)
        self.update_visualisation()
        self.viewer.refresh_inspector()  # Refresh full inspector if open
        self.bottom.quick_inspector.refresh_objects()  # Refresh quick inspector
        self._log(message)

    def _on_channels_changed(self) -> None:
        self.update_visualisation()

    def _on_fluor_selection_changed(self) -> None:
        self.update_visualisation()

    def update_visualisation(self) -> None:
        if not self.state.data.has_data:
            self.viewer.composite_view.update(self.state, [])
            self.bottom.update_views([])
            return
        active_channels = self.viewer.active_channel_flags()
        if not active_channels:
            active_channels = [True] * self.state.data.Y.shape[0]
        self.viewer.composite_view.update(self.state, active_channels)
        self.bottom.update_views(active_channels)

    # ------------------------------------------------------------------
    # Expand helpers

    def _expand_composite_view(self) -> None:
        self.viewer.composite_view.show_expanded(self)

    def _expand_spectral_view(self) -> None:
        self.bottom.spectral_panel.show_expanded(self)

    def _expand_abundance_view(self) -> None:
        self.bottom.abundance_panel.show_expanded(self, self.state)

    # ------------------------------------------------------------------
    # Utilities

    def _show_startup_message(self) -> None:
        self._log('=== Spectral Visualization Playground ===')
        if self.save_root:
            rel = os.path.relpath(self.save_root, os.getcwd())
            self._log(f'Save directory: {rel}')
        self._log('Ready to visualise spectral data!')
        self._log('')

    def _setup_save_directory(self) -> str:
        base = os.path.join(os.getcwd(), 'saved_data')
        subdirs = ['datasets', 'images', 'plots', 'exports']
        try:
            for sub in subdirs:
                Path(base, sub).mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover
            self._log(f'Warning: could not prepare save directory: {exc}')
            return os.getcwd()
        return base

    def _default_path(self, subdir: str) -> str:
        base = self.save_root or os.getcwd()
        path = Path(base) / subdir
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _log(self, message: str) -> None:
        self.bottom.log(message)
    
    def _open_full_inspector(self, object_ids: list = None) -> None:
        """Open the full inspector window.
        
        Args:
            object_ids: Optional list of object IDs to select when opening
        """
        self.viewer._open_inspector(object_ids)
    
    def _get_fluorophore_names(self) -> list:
        """Get list of fluorophore names from current dataset."""
        if self.state.data and self.state.data.spectral:
            return [f.name for f in self.state.data.spectral.fluors]
        return []
    
    def _on_composite_object_selection(self, object_ids: list) -> None:
        """Handle object selection from composite view."""
        # Update QuickInspector
        self.bottom.quick_inspector.set_selection(object_ids)


def main() -> None:
    app = PlaygroundGUI()
    app.mainloop()


if __name__ == '__main__':
    main()
