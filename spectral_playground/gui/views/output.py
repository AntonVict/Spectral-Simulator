from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class OutputPanel(ttk.Frame):
    """Simple text output panel for logging actions."""

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        control = ttk.Frame(self)
        control.pack(fill=tk.X, padx=4, pady=2)

        self._toggle_var = tk.BooleanVar(value=True)
        self.toggle_button = ttk.Button(control, text='Hide Log', command=self._toggle)
        self.toggle_button.pack(side=tk.RIGHT)

        self.container = ttk.Frame(self)
        self.container.pack(fill=tk.BOTH, expand=True)

        self.text = tk.Text(self.container, height=6)
        scrollbar = ttk.Scrollbar(self.container, orient='vertical', command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)

        self.text.pack(side='left', fill='both', expand=True, padx=4, pady=4)
        scrollbar.pack(side='right', fill='y', pady=4)

    def log(self, message: str) -> None:
        self.text.insert(tk.END, message + '\n')
        self.text.see(tk.END)

    def clear(self) -> None:
        self.text.delete('1.0', tk.END)

    def _toggle(self) -> None:
        visible = self._toggle_var.get()
        if visible:
            self.container.forget()
            self.toggle_button.config(text='Show Log')
        else:
            self.container.pack(fill=tk.BOTH, expand=True)
            self.toggle_button.config(text='Hide Log')
        self._toggle_var.set(not visible)
