"""Simplified channel manager for detection channels."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, List, Dict, Any, Tuple, Optional
import numpy as np


class ChannelListManager:
    """Manages detection channels with compact UI."""
    
    def __init__(
        self, 
        parent: tk.Widget,
        log_callback: Callable[[str], None],
        get_wavelength_range_callback: Callable[[], Tuple[float, float]]
    ):
        """Initialize channel manager."""
        self.parent = parent
        self.log = log_callback
        self.get_wavelength_range = get_wavelength_range_callback
        self.channel_vars: List[Dict[str, tk.Variable]] = []
        
        self._build_ui()
        
        # Initialize with 4 default channels
        self._initialize_default_channels()
    
    def _build_ui(self) -> None:
        """Build compact UI."""
        # Controls at top
        controls_frame = ttk.Frame(self.parent)
        controls_frame.pack(fill=tk.X, padx=2, pady=2)
        
        ttk.Button(
            controls_frame,
            text='+',
            command=self._add_channel,
            width=3
        ).pack(side=tk.LEFT, padx=(0, 2))
        
        ttk.Button(
            controls_frame,
            text='Auto-Space',
            command=self._auto_space_channels,
            width=10
        ).pack(side=tk.LEFT, padx=2)
        
        # Scrollable channel list
        self.list_frame = ttk.Frame(self.parent)
        self.list_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
    
    def _initialize_default_channels(self) -> None:
        """Initialize with 4 evenly-spaced channels."""
        try:
            wl_min, wl_max = self.get_wavelength_range()
            bandwidth = 30.0
            
            centers = np.linspace(
                wl_min + 0.5 * bandwidth,
                wl_max - 0.5 * bandwidth,
                4
            )
            
            for idx, center in enumerate(centers):
                self._add_channel_internal(f'C{idx + 1}', float(center), bandwidth)
        except Exception as e:
            self.log(f'Error initializing channels: {e}')
    
    def _add_channel(self) -> None:
        """Add a new channel."""
        try:
            wl_min, wl_max = self.get_wavelength_range()
            center = (wl_min + wl_max) / 2
        except:
            center = 500.0
        
        idx = len(self.channel_vars)
        self._add_channel_internal(f'C{idx + 1}', center, 30.0)
        self.log(f'Added channel C{idx + 1}')
    
    def _add_channel_internal(self, name: str, center: float, bandwidth: float) -> None:
        """Internal method to add a channel."""
        idx = len(self.channel_vars)
        
        # Create frame for this channel
        channel_frame = ttk.Frame(self.list_frame)
        channel_frame.pack(fill=tk.X, pady=1)
        
        # Variables
        name_var = tk.StringVar(value=name)
        center_var = tk.DoubleVar(value=center)
        bandwidth_var = tk.DoubleVar(value=bandwidth)
        
        # Widgets
        ttk.Entry(channel_frame, textvariable=name_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Entry(channel_frame, textvariable=center_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Entry(channel_frame, textvariable=bandwidth_var, width=8).pack(side=tk.LEFT, padx=2)
        
        # Delete button
        ttk.Button(
            channel_frame,
            text='×',
            command=lambda: self._delete_channel(idx),
            width=3
        ).pack(side=tk.LEFT, padx=2)
        
        # Store references
        self.channel_vars.append({
            'frame': channel_frame,
            'name': name_var,
            'center': center_var,
            'bandwidth': bandwidth_var
        })
    
    def _delete_channel(self, idx: int) -> None:
        """Delete a channel."""
        if len(self.channel_vars) <= 1:
            self.log('At least one channel is required')
            return
        
        if 0 <= idx < len(self.channel_vars):
            self.channel_vars[idx]['frame'].destroy()
            self.channel_vars.pop(idx)
            self._rebuild_list()
            self.log(f'Deleted channel at position {idx + 1}')
    
    def _auto_space_channels(self) -> None:
        """Evenly space all channels across the wavelength range."""
        try:
            wl_min, wl_max = self.get_wavelength_range()
            n_channels = len(self.channel_vars)
            
            if n_channels == 0:
                return
            
            # Get average bandwidth
            bandwidths = [ch['bandwidth'].get() for ch in self.channel_vars]
            avg_bandwidth = np.mean(bandwidths)
            
            # Calculate evenly-spaced centers
            centers = np.linspace(
                wl_min + 0.5 * avg_bandwidth,
                wl_max - 0.5 * avg_bandwidth,
                n_channels
            )
            
            # Update each channel
            for idx, (channel_vars, center) in enumerate(zip(self.channel_vars, centers)):
                channel_vars['center'].set(float(center))
                channel_vars['name'].set(f'C{idx + 1}')
            
            self.log(f'Auto-spaced {n_channels} channels')
            
        except Exception as e:
            self.log(f'Error auto-spacing channels: {e}')
    
    def _rebuild_list(self) -> None:
        """Rebuild the channel list after deletion."""
        # Destroy all frames
        for widget in self.list_frame.winfo_children():
            widget.destroy()
        
        # Recreate all channels
        temp_vars = self.channel_vars.copy()
        self.channel_vars.clear()
        
        for channel_data in temp_vars:
            if channel_data['frame'].winfo_exists():
                continue
            self._add_channel_internal(
                channel_data['name'].get(),
                channel_data['center'].get(),
                channel_data['bandwidth'].get()
            )
    
    def get_all_channels(self) -> List[Dict[str, Any]]:
        """Get configuration for all channels."""
        channels = []
        for ch in self.channel_vars:
            try:
                channels.append({
                    'name': ch['name'].get(),
                    'center_nm': float(ch['center'].get()),
                    'bandwidth_nm': float(ch['bandwidth'].get())
                })
            except (ValueError, tk.TclError):
                # Skip invalid channels
                pass
        return channels
    
    def set_channels(self, channels: List[Dict[str, Any]]) -> None:
        """Set channels from configuration (for loading)."""
        # Clear existing channels
        for ch in self.channel_vars:
            ch['frame'].destroy()
        self.channel_vars.clear()
        
        # Add new channels
        for channel_data in channels:
            self._add_channel_internal(
                channel_data.get('name', 'C1'),
                channel_data.get('center_nm', 500.0),
                channel_data.get('bandwidth_nm', 30.0)
            )
        
        self.log(f'Loaded {len(channels)} channel(s)')
    
    def validate_all(self) -> Tuple[bool, Optional[str]]:
        """Validate all channels."""
        try:
            wl_min, wl_max = self.get_wavelength_range()
            channels = self.get_all_channels()
            
            if not channels:
                return False, "At least one channel is required"
            
            for ch in channels:
                center = ch['center_nm']
                bandwidth = ch['bandwidth_nm']
                
                # Check bandwidth is positive
                if bandwidth <= 0:
                    return False, f"{ch['name']}: Bandwidth must be positive"
                
                # Check center is within grid
                if center < wl_min or center > wl_max:
                    return False, f"{ch['name']}: Center ({center:.1f} nm) outside wavelength range ({wl_min:.1f}-{wl_max:.1f} nm)"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {e}"
