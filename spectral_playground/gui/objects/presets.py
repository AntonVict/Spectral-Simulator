"""Preset object generation utilities."""

from __future__ import annotations
import random
from typing import List, Dict, Any, Tuple


class PresetGenerator:
    """Generates preset and quick-assign object configurations."""
    
    @staticmethod
    def generate_default_presets(img_h: int, img_w: int) -> List[Dict[str, Any]]:
        """Generate 3 preset objects, one for each default fluorophore.
        
        Args:
            img_h: Image height in pixels
            img_w: Image width in pixels
            
        Returns:
            List of preset object dictionaries
        """
        presets = [
            {
                'fluor_index': 0,
                'kind': 'gaussian_blobs',
                'region': {'type': 'full'},
                'count': 30,
                'size_px': 4.0,
                'intensity_min': 0.8,
                'intensity_max': 1.2,
                'spot_sigma': 3.0,  # Larger sigma for visible Gaussian blobs
            },
            {
                'fluor_index': 1,
                'kind': 'circles',
                'region': {'type': 'full'},
                'count': 20,
                'size_px': 6.0,
                'intensity_min': 0.6,
                'intensity_max': 1.0,
                'spot_sigma': 1.5,  # Not used for circles, but set for consistency
            },
            {
                'fluor_index': 2,
                'kind': 'dots',
                'region': {'type': 'full'},
                'count': 40,
                'size_px': 3.0,
                'intensity_min': 0.4,
                'intensity_max': 0.8,
                'spot_sigma': 1.5,  # Moderate sigma for dot-like appearance
            }
        ]
        
        return presets
    
    @staticmethod
    def generate_quick_assign_sample(
        img_h: int,
        img_w: int,
        fluorophore_names: List[str]
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Generate a sample dataset with all available fluorophores.
        
        Creates a varied distribution of Gaussian blobs with:
        - All available fluorophores
        - Varying spot counts (100-2000) across fluorophores
        - Random positions
        - Random amplitudes (0.5-1.5)
        - Random spot sigma (0.8-2.5) - TIGHTER RANGE for reasonable sizes
        - One small contamination dot with low intensity
        
        Args:
            img_h: Image height in pixels
            img_w: Image width in pixels
            fluorophore_names: List of fluorophore names
            
        Returns:
            Tuple of (objects list, log message)
        """
        if not fluorophore_names:
            return [], "No fluorophores available"
        
        num_fluors = len(fluorophore_names)
        objects = []
        
        # Generate spot counts for each fluorophore
        total_spots = 0
        for fluor_idx, fluor_name in enumerate(fluorophore_names):
            # Random count between 100-2000 for each fluorophore
            count = random.randint(100, 2000)
            
            # Add one object that generates 'count' spots
            amplitude_min = 0.5
            amplitude_max = 1.5
            # FIXED: Tighter sigma range (0.8-2.5 instead of 0.5-6.0)
            # This gives effective radius of 1.6-5.0 pixels instead of 1.0-12.0
            spot_sigma = random.uniform(0.8, 2.5)
            
            obj = {
                'fluor_index': fluor_idx,
                'kind': 'gaussian_blobs',
                'region': {'type': 'full'},
                'count': count,
                'size_px': 0,  # Not used for Gaussian blobs
                'intensity_min': amplitude_min,
                'intensity_max': amplitude_max,
                'spot_sigma': spot_sigma
            }
            objects.append(obj)
            
            total_spots += count
        
        # Add contamination dot (small, low intensity, random fluorophore)
        contam_fluor_idx = random.randint(0, num_fluors - 1)
        contam_fluor_name = fluorophore_names[contam_fluor_idx]
        contam_sigma = random.uniform(0.3, 0.8)
        
        contam_obj = {
            'fluor_index': contam_fluor_idx,
            'kind': 'dots',
            'region': {'type': 'full'},
            'count': 1,
            'size_px': 0,
            'intensity_min': 0.1,
            'intensity_max': 0.3,
            'spot_sigma': contam_sigma
        }
        objects.append(contam_obj)
        
        # Create summary message
        log_msg = f"Generated {total_spots + 1} spots across {num_fluors} fluorophores\n"
        log_msg += f"   + 1 contamination dot ({contam_fluor_name}, σ={contam_sigma:.2f})"
        
        return objects, log_msg

