"""Preset object generation utilities."""

from __future__ import annotations
import random
from typing import List, Dict, Any, Tuple
import numpy as np


class PresetGenerator:
    """Generates preset and quick-assign object configurations."""
    
    @staticmethod
    def generate_default_presets(img_h: int, img_w: int) -> List[Dict[str, Any]]:
        """Generate default preset objects (empty for now).
        
        Args:
            img_h: Image height in pixels
            img_w: Image width in pixels
            
        Returns:
            Empty list - use quick-assign for multi-fluorophore objects
        """
        # Return empty - users should use quick-assign multi-fluorophore
        return []
    
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
    
    @staticmethod
    def quick_assign_multi_fluorophore(
        num_fluors: int,
        num_continuous: int = None,
        continuous_indices: List[int] = None,
        binary_indices: List[int] = None,
        use_dirichlet: bool = True,
        img_h: int = 128,
        img_w: int = 128
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Create one object spec per binary fluorophore.
        
        If 7 fluorophores with continuous=[0,1] and binary=[2,3,4,5,6]:
        - Creates 5 object specs (O1-O5)
        - Each has F1+F2 at random ratios + ONE binary (F3, F4, F5, F6, or F7)
        
        Args:
            num_fluors: Total fluorophores available
            num_continuous: DEPRECATED - use continuous_indices instead
            continuous_indices: List of fluorophore indices for continuous ratios
            binary_indices: List of fluorophore indices to choose binary from
            use_dirichlet: Use Dirichlet distribution for ratios
            img_h: Image height in pixels
            img_w: Image width in pixels
            
        Returns:
            Tuple of (objects list, log message)
        """
        from .composition import CompositionGenerator
        
        # Handle backward compatibility
        if continuous_indices is None:
            if num_continuous is not None and num_continuous > 0:
                continuous_indices = list(range(num_continuous))
            else:
                continuous_indices = []
        
        if binary_indices is None:
            if num_continuous is not None:
                binary_indices = list(range(num_continuous, num_fluors))
            else:
                binary_indices = [i for i in range(num_fluors) if i not in continuous_indices]
        
        if not binary_indices:
            return [], "Need at least one binary fluorophore"
        
        objects = []
        
        for obj_num, binary_idx in enumerate(binary_indices, start=1):
            # Generate composition for this binary fluorophore
            comp_data = CompositionGenerator.generate_composition(
                num_total_fluors=num_fluors,
                continuous_indices=continuous_indices,
                binary_indices=[binary_idx],  # Force this specific binary
                use_dirichlet=use_dirichlet
            )
            
            # Random parameters
            sigma = random.uniform(0.8, 2.5)
            radius = 2.0 * sigma
            count = random.randint(500, 2000)  # 10x increase for more realistic datasets
            
            obj = {
                'name': f'O{obj_num}',
                'composition': comp_data['composition'],
                'binary_fluor': comp_data['binary_fluor'],
                'kind': 'gaussian_blobs',
                'region': {'type': 'full'},
                'count': count,
                'spot_sigma': sigma,
                'radius': radius,
                'intensity_min': 0.5,
                'intensity_max': 1.5,
                'size_px': 0,  # Not used for Gaussian blobs
            }
            objects.append(obj)
        
        log_msg = f"Quick-assigned {len(objects)} multi-fluorophore objects\n"
        log_msg += f"Each has {len(continuous_indices)} continuous + 1 binary fluorophore"
        
        return objects, log_msg
    
    @staticmethod
    def quick_assign_single_fluorophore(
        num_fluors: int,
        img_h: int = 128,
        img_w: int = 128
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Create one single-fluorophore object spec per fluorophore.
        
        If 5 fluorophores:
        - Creates 5 object specs (O1-O5)
        - O1 uses F1 @ 100%, O2 uses F2 @ 100%, etc.
        
        Args:
            num_fluors: Total fluorophores available
            img_h: Image height in pixels
            img_w: Image width in pixels
            
        Returns:
            Tuple of (objects list, log message)
        """
        if num_fluors == 0:
            return [], "No fluorophores available"
        
        objects = []
        
        for fluor_idx in range(num_fluors):
            # Random parameters
            sigma = random.uniform(0.8, 2.5)
            radius = 2.0 * sigma
            count = random.randint(500, 2000)  # Match multi-fluorophore counts
            
            obj = {
                'name': f'O{fluor_idx + 1}',
                'mode': 'single',  # Single-fluorophore mode
                'composition': [{'fluor_index': fluor_idx, 'ratio': 1.0}],
                'binary_fluor': None,
                'kind': 'gaussian_blobs',
                'region': {'type': 'full'},
                'count': count,
                'spot_sigma': sigma,
                'radius': radius,
                'intensity_min': 0.5,
                'intensity_max': 1.5,
                'size_px': 0,  # Not used for Gaussian blobs
            }
            objects.append(obj)
        
        log_msg = f"Quick-assigned {len(objects)} single-fluorophore objects\n"
        log_msg += f"One object per fluorophore (F1-F{num_fluors})"
        
        return objects, log_msg

