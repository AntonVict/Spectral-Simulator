"""RGB rendering from spectral data."""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ....state import PlaygroundData

from ...utils import wavelength_to_rgb_nm


class RGBRenderer:
    """Handles RGB composition from spectral data."""
    
    def __init__(self, visual_settings_manager):
        """Initialize RGB renderer.
        
        Args:
            visual_settings_manager: Visual settings manager for rendering parameters
        """
        self.visual_settings = visual_settings_manager
    
    def render_composite(self, data: 'PlaygroundData', channels: list) -> np.ndarray:
        """Render the composite RGB image from spectral data.
        
        Args:
            data: PlaygroundData object
            channels: List of active channel flags
            
        Returns:
            RGB image array (H, W, 3)
        """
        H, W = data.field.shape
        Y = data.Y
        
        if not any(channels):
            channels = [True] * Y.shape[0]

        rgb = np.zeros((H, W, 3), dtype=np.float32)
        eps = 1e-6
        
        # Get intensity mapping settings
        settings = self.visual_settings
        percentile = settings.percentile_threshold.get()
        gamma = settings.gamma_correction.get()
        use_log = settings.use_log_scaling.get()
        
        # Calculate normalization scale based on mode
        if settings.normalization_mode.get() == "global":
            global_max = self._calculate_global_max(Y, channels, percentile, H, W)
            global_scale = global_max + eps
        
        # Process each channel
        for idx, flag in enumerate(channels):
            if not flag:
                continue
            
            channel_image = Y[idx].reshape(H, W)
            
            # Determine normalization scale
            if settings.normalization_mode.get() == "global":
                scale = global_scale
            else:  # per_channel (default)
                if percentile >= 100.0:
                    scale = np.max(channel_image) + eps
                else:
                    scale = np.percentile(channel_image, percentile) + eps
            
            # Apply logarithmic scaling if enabled
            if use_log:
                normalized = np.log1p(channel_image) / np.log1p(scale)
            else:
                normalized = channel_image / scale
            
            # Apply gamma correction
            if gamma != 1.0:
                normalized = np.power(np.clip(normalized, 0.0, 1.0), gamma)
            
            # Final clipping
            normalized = np.clip(normalized, 0.0, 1.0)
            
            # Map to color
            color = np.array(wavelength_to_rgb_nm(data.spectral.channels[idx].center_nm), dtype=np.float32)
            rgb += normalized[..., None] * color[None, None, :]

        return np.clip(rgb, 0.0, 1.0)
    
    def _calculate_global_max(self, Y: np.ndarray, channels: list, percentile: float, H: int, W: int) -> float:
        """Calculate the global maximum across all active channels.
        
        Args:
            Y: Spectral data array (L, P)
            channels: List of active channel flags
            percentile: Percentile threshold for normalization
            H: Image height
            W: Image width
            
        Returns:
            Global maximum value
        """
        global_max = 0.0
        for idx, flag in enumerate(channels):
            if flag:
                channel_image = Y[idx].reshape(H, W)
                if percentile >= 100.0:
                    channel_max = np.max(channel_image)
                else:
                    channel_max = np.percentile(channel_image, percentile)
                global_max = max(global_max, channel_max)
        return global_max

