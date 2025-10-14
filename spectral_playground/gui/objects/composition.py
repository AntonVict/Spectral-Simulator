"""Multi-fluorophore composition generation."""

import numpy as np
from typing import List, Dict, Any, Optional


class CompositionGenerator:
    """Generate multi-fluorophore compositions with continuous + binary."""
    
    @staticmethod
    def generate_composition(
        num_total_fluors: int,
        num_continuous: int = None,
        continuous_indices: List[int] = None,
        binary_indices: List[int] = None,
        use_dirichlet: bool = True,
        rng: Optional[np.random.Generator] = None
    ) -> Dict[str, Any]:
        """Generate random composition with explicit fluorophore role assignments.
        
        Args:
            num_total_fluors: Total fluorophores available
            num_continuous: DEPRECATED - use continuous_indices instead
            continuous_indices: List of fluorophore indices to use for continuous ratios (e.g., [0, 1])
            binary_indices: List of fluorophore indices to choose binary from (e.g., [2, 3, 4])
            use_dirichlet: Use Dirichlet distribution vs uniform random
            rng: Random number generator
            
        Returns:
            {
                'composition': [
                    {'fluor_index': 0, 'ratio': 0.45},
                    {'fluor_index': 1, 'ratio': 0.55},
                    {'fluor_index': 3, 'ratio': 1.0}
                ],
                'binary_fluor': 3
            }
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Handle backward compatibility
        if continuous_indices is None:
            if num_continuous is not None and num_continuous > 0:
                continuous_indices = list(range(num_continuous))
            else:
                continuous_indices = []
        
        if binary_indices is None:
            if num_continuous is not None:
                binary_indices = list(range(num_continuous, num_total_fluors))
            else:
                # Default: all non-continuous fluorophores are binary
                binary_indices = [i for i in range(num_total_fluors) if i not in continuous_indices]
        
        composition = []
        
        # Generate continuous fluorophore ratios
        if continuous_indices:
            num_cont = len(continuous_indices)
            if use_dirichlet:
                # Dirichlet distribution for normalized ratios
                alpha = np.ones(num_cont)
                ratios = rng.dirichlet(alpha)
            else:
                # Uniform random, then normalize
                ratios = rng.uniform(0.3, 0.7, num_cont)
                ratios = ratios / ratios.sum()
            
            for i, fluor_idx in enumerate(continuous_indices):
                composition.append({
                    'fluor_index': fluor_idx,
                    'ratio': float(ratios[i])
                })
        
        # Pick one binary fluorophore from binary_indices
        if binary_indices:
            binary_fluor = int(rng.choice(binary_indices))
            composition.append({
                'fluor_index': binary_fluor,
                'ratio': 1.0
            })
        else:
            binary_fluor = None
        
        return {
            'composition': composition,
            'binary_fluor': binary_fluor
        }
    
    @staticmethod
    def composition_to_display(
        composition: List[Dict], 
        binary_fluor: Optional[int] = None,
        fluor_names: Optional[List[str]] = None
    ) -> str:
        """Format composition for display.
        
        Args:
            composition: List of {'fluor_index': k, 'ratio': r}
            binary_fluor: Index of the binary fluorophore (if any)
            fluor_names: List of actual fluorophore names (e.g., ["DAPI", "GFP", ...])
        
        Returns: "DAPI+GFP+RFP" (binary first, then continuous)
        """
        if not composition:
            return "Empty"
        
        # Sort: binary first, then continuous by index
        binary_parts = []
        continuous_parts = []
        
        for c in composition:
            fluor_idx = c['fluor_index']
            # Use actual fluorophore name if available, otherwise fall back to F{idx+1}
            if fluor_names and 0 <= fluor_idx < len(fluor_names):
                fluor_name = fluor_names[fluor_idx]
            else:
                fluor_name = f"F{fluor_idx + 1}"
            
            if fluor_idx == binary_fluor:
                binary_parts.append(fluor_name)
            else:
                continuous_parts.append(fluor_name)
        
        # Combine: binary + continuous
        all_parts = binary_parts + sorted(continuous_parts)
        return "+".join(all_parts)

