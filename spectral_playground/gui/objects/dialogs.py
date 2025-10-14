"""Template management dialog for multi-fluorophore compositions."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, List
from .templates import TemplateManager, ObjectTemplate, FluorophoreComponent


class TemplateEditorDialog:
    """Dialog for managing multi-fluorophore templates."""
    
    def __init__(
        self,
        parent: tk.Widget,
        template_manager: TemplateManager,
        fluorophore_index_to_name: Callable[[int], str],
        fluorophore_name_to_index: Callable[[str], int],
        get_fluorophore_list: Callable[[], List[str]],
        log_callback: Callable[[str], None]
    ):
        """Initialize template editor dialog.
        
        Args:
            parent: Parent widget
            template_manager: Template manager instance
            fluorophore_index_to_name: Function to convert index to name
            fluorophore_name_to_index: Function to convert name to index
            get_fluorophore_list: Function to get list of fluorophore names
            log_callback: Logging function
        """
        self.parent = parent
        self.template_manager = template_manager
        self.fluorophore_index_to_name = fluorophore_index_to_name
        self.fluorophore_name_to_index = fluorophore_name_to_index
        self.get_fluorophore_list = get_fluorophore_list
        self.log = log_callback
        self.dialog = None
        
    def show(self):
        """Show the template manager dialog."""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Manage Multi-Fluorophore Templates")
        self.dialog.geometry("700x550")
        self.dialog.transient(self.parent)
        
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Template list
        list_frame = ttk.LabelFrame(main_frame, text="Available Templates")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0,10))
        
        # Listbox with scrollbar
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        template_listbox = tk.Listbox(list_container, yscrollcommand=scrollbar.set, height=10)
        template_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=template_listbox.yview)
        
        # Populate template list
        def refresh_list():
            template_listbox.delete(0, tk.END)
            for template in self.template_manager.templates:
                # Show composition summary
                comp_summary = ", ".join([f"F{c.fluor_index+1}:{c.ratio:.0%}" 
                                         for c in template.composition])
                template_listbox.insert(tk.END, f"{template.name} ({comp_summary})")
        
        refresh_list()
        
        # Template details frame
        detail_frame = ttk.LabelFrame(main_frame, text="Template Details")
        detail_frame.pack(fill=tk.X, pady=(0,10))
        
        detail_text = tk.Text(detail_frame, height=6, width=60, state='disabled')
        detail_text.pack(padx=5, pady=5)
        
        def show_template_details(event=None):
            """Show details of selected template."""
            selection = template_listbox.curselection()
            if not selection:
                return
            
            idx = selection[0]
            if idx >= len(self.template_manager.templates):
                return
            
            template = self.template_manager.templates[idx]
            
            details = f"Name: {template.name}\n"
            details += f"Description: {template.description}\n\n"
            details += "Composition:\n"
            for comp in template.composition:
                fluor_name = self.fluorophore_index_to_name(comp.fluor_index)
                details += f"  • {fluor_name}: {comp.ratio:.1%} "
                if comp.ratio_noise > 0:
                    details += f"(±{comp.ratio_noise:.1%} noise)"
                details += "\n"
            
            detail_text.config(state='normal')
            detail_text.delete('1.0', tk.END)
            detail_text.insert('1.0', details)
            detail_text.config(state='disabled')
        
        template_listbox.bind('<<ListboxSelect>>', show_template_details)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Create New", 
                  command=lambda: self._create_template_dialog(refresh_list)).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Delete Selected", 
                  command=lambda: self._delete_template(template_listbox, refresh_list)).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Close", command=self.dialog.destroy).pack(side=tk.RIGHT)
    
    def _create_template_dialog(self, refresh_callback: Callable[[], None]):
        """Open dialog to create a custom template."""
        dialog = tk.Toplevel(self.dialog)
        dialog.title("Create Custom Template")
        dialog.geometry("500x450")
        dialog.transient(self.dialog)
        
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Template name
        ttk.Label(main_frame, text="Template Name:").grid(row=0, column=0, sticky='w', pady=5)
        name_var = tk.StringVar(value="Custom Mix")
        ttk.Entry(main_frame, textvariable=name_var, width=30).grid(row=0, column=1, sticky='w', pady=5)
        
        # Description
        ttk.Label(main_frame, text="Description:").grid(row=1, column=0, sticky='w', pady=5)
        desc_var = tk.StringVar(value="")
        ttk.Entry(main_frame, textvariable=desc_var, width=30).grid(row=1, column=1, sticky='w', pady=5)
        
        # Composition builder
        comp_frame = ttk.LabelFrame(main_frame, text="Fluorophore Composition")
        comp_frame.grid(row=2, column=0, columnspan=2, sticky='ew', pady=10)
        
        # List of components
        components = []  # List of (fluor_index, ratio, ratio_noise)
        
        comp_listbox = tk.Listbox(comp_frame, height=6)
        comp_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        def refresh_comp_list():
            comp_listbox.delete(0, tk.END)
            total_ratio = sum(c[1] for c in components)
            for fluor_idx, ratio, noise in components:
                fluor_name = self.fluorophore_index_to_name(fluor_idx)
                comp_listbox.insert(tk.END, 
                                  f"{fluor_name}: {ratio:.1%} ±{noise:.1%} (Total: {total_ratio:.1%})")
        
        # Add component controls
        add_frame = ttk.Frame(comp_frame)
        add_frame.pack(fill=tk.X, padx=5, pady=5)
        
        fluor_names = self.get_fluorophore_list()
        fluor_var = tk.StringVar(value=fluor_names[0] if fluor_names else "F1")
        ratio_var = tk.DoubleVar(value=0.5)
        noise_var = tk.DoubleVar(value=0.05)
        
        ttk.Label(add_frame, text="Fluorophore:").grid(row=0, column=0, sticky='w', padx=2)
        ttk.Combobox(add_frame, textvariable=fluor_var, values=fluor_names, 
                    state='readonly', width=10).grid(row=0, column=1, padx=2)
        
        ttk.Label(add_frame, text="Ratio:").grid(row=0, column=2, sticky='w', padx=2)
        ttk.Entry(add_frame, textvariable=ratio_var, width=8).grid(row=0, column=3, padx=2)
        
        ttk.Label(add_frame, text="Noise:").grid(row=1, column=0, sticky='w', padx=2)
        ttk.Entry(add_frame, textvariable=noise_var, width=8).grid(row=1, column=1, padx=2)
        
        def add_component():
            fluor_idx = self.fluorophore_name_to_index(fluor_var.get())
            components.append((fluor_idx, ratio_var.get(), noise_var.get()))
            refresh_comp_list()
        
        def remove_component():
            selection = comp_listbox.curselection()
            if selection and components:
                components.pop(selection[0])
                refresh_comp_list()
        
        ttk.Button(add_frame, text="Add", command=add_component).grid(row=1, column=2, padx=2)
        ttk.Button(add_frame, text="Remove", command=remove_component).grid(row=1, column=3, padx=2)
        
        # Save button
        def save_template():
            if not components:
                messagebox.showwarning("No Components", "Add at least one fluorophore component!")
                return
            
            template = ObjectTemplate(
                name=name_var.get(),
                description=desc_var.get(),
                composition=[FluorophoreComponent(f_idx, ratio, noise) 
                            for f_idx, ratio, noise in components]
            )
            
            self.template_manager.add_template(template)
            refresh_callback()
            self.log(f"Created template: {template.name}")
            dialog.destroy()
        
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        ttk.Button(button_frame, text="Save Template", command=save_template).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT)
    
    def _delete_template(self, listbox: tk.Listbox, refresh_callback: Callable[[], None]):
        """Delete selected template."""
        selection = listbox.curselection()
        if not selection:
            return
        
        idx = selection[0]
        if idx >= len(self.template_manager.templates):
            return
        
        template = self.template_manager.templates[idx]
        
        # Don't allow deleting built-in single fluorophore templates
        if template.name.endswith(" Only"):
            messagebox.showinfo("Cannot Delete", "Built-in templates cannot be deleted.")
            return
        
        if messagebox.askyesno("Confirm Delete", f"Delete template '{template.name}'?"):
            self.template_manager.remove_template(template.name)
            refresh_callback()
            self.log(f"Deleted template: {template.name}")

