"""Auto-fit utilities for extracting parameters from scenes."""

from __future__ import annotations

from typing import Dict, Any, Callable
from tkinter import messagebox

import numpy as np


def auto_fit_parameters_from_scene(
    scene,
    log_callback: Callable[[str], None]
) -> Dict[str, Any]:
    """Auto-fit statistical parameters from geometric scene.
    
    Extracts spatial intensity (λ) and radius distribution parameters
    from the objects in a scene.
    
    Args:
        scene: GeometricScene containing objects
        log_callback: Function to log messages
        
    Returns:
        Dictionary with fitted parameters:
        - lambda_val: Spatial intensity (objects per pixel²)
        - radius_mean: Mean radius
        - radius_std: Standard deviation of radius
        - radius_min: Minimum radius
        - radius_max: Maximum radius
        - status_message: Formatted status string
        - info_message: Detailed info message for display
        
    Raises:
        ValueError: If scene has no objects or no radii
    """
    if scene is None or len(scene) == 0:
        raise ValueError('No objects in current scene.')
    
    # Extract radii from scene
    radii = [obj.radius for obj in scene.objects]
    
    if len(radii) == 0:
        raise ValueError('No radii found in objects.')
    
    # Compute radius statistics
    mean_r = float(np.mean(radii))
    std_r = float(np.std(radii))
    min_r = float(max(1.0, min(radii)))
    max_r = float(max(radii))
    
    # Compute intensity (λ) from scene
    H, W = scene.field_shape
    λ_empirical = len(scene) / (H * W)
    
    # Create status message
    status_message = f'✓ Fitted: λ={λ_empirical:.6f}, μ_R={mean_r:.2f}, σ_R={std_r:.2f}'
    
    # Create detailed info message
    info_message = (
        f'All parameters fitted from scene:\n\n'
        f'Spatial density (λ): {λ_empirical:.6f} obj/px²\n'
        f'Radius mean (μ): {mean_r:.2f} px\n'
        f'Radius std (σ): {std_r:.2f} px\n'
        f'Radius range: [{min_r:.2f}, {max_r:.2f}] px'
    )
    
    log_callback(f'Auto-fit complete: λ={λ_empirical:.6f}, μ={mean_r:.2f}, σ={std_r:.2f}')
    
    return {
        'lambda_val': λ_empirical,
        'radius_mean': mean_r,
        'radius_std': std_r,
        'radius_min': min_r,
        'radius_max': max_r,
        'status_message': status_message,
        'info_message': info_message
    }

