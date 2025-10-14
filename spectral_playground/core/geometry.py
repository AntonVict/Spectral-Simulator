"""Geometric object representation for statistical analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


Array = np.ndarray


@dataclass
class GeometricObject:
    """Single geometric object with full spatial information.
    
    Attributes:
        id: Unique identifier
        position: (y, x) coordinates in pixels
        radius: Effective radius in pixels
        composition: List of {'fluor_index': k, 'ratio': r, 'intensity': i}
        type: Object type ('gaussian_blobs', 'circles', 'boxes', 'dots')
        spot_sigma: Sigma parameter for Gaussian rendering
        base_intensity: Base amplitude/intensity of the object
        size_px: Original size parameter in pixels
        source_spec_index: Index of the source object specification (for lookup)
    """
    id: int
    position: Tuple[float, float]
    radius: float
    composition: List[Dict[str, float]]
    type: str
    spot_sigma: float = 2.0
    base_intensity: float = 1.0
    size_px: float = 3.0
    source_spec_index: int = -1


class GeometricScene:
    """Container for geometric objects with spatial indexing and overlap computation.
    
    This class manages a collection of geometric objects and provides efficient
    spatial queries, overlap detection, and conversion to various formats.
    """
    
    def __init__(self, objects: List[GeometricObject], field_shape: Tuple[int, int]):
        """Initialize geometric scene.
        
        Args:
            objects: List of GeometricObject instances
            field_shape: (H, W) dimensions of the field in pixels
        """
        self.objects = objects
        self.field_shape = field_shape
        
        # Lazy-loaded properties
        self._spatial_index: Optional[cKDTree] = None
        self._overlap_graph: Optional[List[List[int]]] = None
        self._dataframe: Optional[pd.DataFrame] = None
    
    @property
    def spatial_index(self) -> cKDTree:
        """Build and cache KD-tree for fast spatial queries."""
        if self._spatial_index is None:
            if len(self.objects) == 0:
                # Empty tree - use dummy point
                centers = np.array([[0.0, 0.0]])
            else:
                centers = np.array([obj.position for obj in self.objects])
            self._spatial_index = cKDTree(centers)
        return self._spatial_index
    
    @property
    def overlap_graph(self) -> List[List[int]]:
        """Get precomputed adjacency list of overlapping object pairs."""
        if self._overlap_graph is None:
            self._precompute_overlaps()
        return self._overlap_graph
    
    def _precompute_overlaps(self) -> None:
        """Precompute all pairwise geometric overlaps using spatial indexing.
        
        Two objects overlap if the distance between their centers is less than
        the sum of their radii.
        
        Optimized with vectorization for large scenes.
        """
        n = len(self.objects)
        self._overlap_graph = [[] for _ in range(n)]
        
        if n == 0:
            return
        
        # Performance guard: warn for very large scenes
        if n > 200000:
            import warnings
            warnings.warn(
                f"Skipping overlap precomputation for {n} objects (would be very slow). "
                "Object-based crowding analysis will be unavailable.",
                UserWarning
            )
            return
        
        # Vectorized data for efficiency
        centers = np.array([obj.position for obj in self.objects])
        radii = np.array([obj.radius for obj in self.objects])
        max_radius = radii.max()
        
        for i, obj in enumerate(self.objects):
            # Query neighbors within maximum possible overlap distance
            search_radius = obj.radius + max_radius
            candidates = self.spatial_index.query_ball_point(
                obj.position,
                r=search_radius
            )
            
            if len(candidates) == 0:
                continue
            
            # Vectorized distance computation for all candidates
            candidate_centers = centers[candidates]
            dists = np.linalg.norm(candidate_centers - np.array(obj.position), axis=1)
            candidate_radii = radii[candidates]
            
            # Check overlap condition for all candidates at once
            overlaps = dists < (obj.radius + candidate_radii)
            
            # Add overlapping pairs (avoiding duplicates and self)
            for idx, j in enumerate(candidates):
                if i >= j:  # Avoid duplicates and self
                    continue
                
                if overlaps[idx]:
                    self._overlap_graph[i].append(j)
                    self._overlap_graph[j].append(i)
    
    def get_neighbors(self, obj_id: int) -> List[int]:
        """Get list of objects that geometrically overlap with given object.
        
        Args:
            obj_id: Object ID to query
            
        Returns:
            List of object IDs that overlap with the given object
        """
        if obj_id < 0 or obj_id >= len(self.objects):
            return []
        return self.overlap_graph[obj_id]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert objects to pandas DataFrame for analysis and export.
        
        Returns:
            DataFrame with columns: id, y, x, radius, type, spot_sigma,
            and fluor_<k>_intensity for each fluorophore
        """
        if self._dataframe is None:
            if len(self.objects) == 0:
                # Return empty DataFrame with correct schema
                self._dataframe = pd.DataFrame(columns=[
                    'id', 'y', 'x', 'radius', 'type', 'spot_sigma'
                ])
            else:
                # Build rows
                rows = []
                for obj in self.objects:
                    row = {
                        'id': obj.id,
                        'y': obj.position[0],
                        'x': obj.position[1],
                        'radius': obj.radius,
                        'type': obj.type,
                        'spot_sigma': obj.spot_sigma,
                    }
                    
                    # Add fluorophore intensities
                    for comp in obj.composition:
                        fluor_idx = comp['fluor_index']
                        intensity = comp['intensity']
                        row[f'fluor_{fluor_idx}_intensity'] = intensity
                        row[f'fluor_{fluor_idx}_ratio'] = comp.get('ratio', 1.0)
                    
                    rows.append(row)
                
                self._dataframe = pd.DataFrame(rows)
        
        return self._dataframe
    
    def rasterize(self, K: int, rng: Optional[np.random.Generator] = None) -> Array:
        """Convert geometric objects to abundance maps for visualization.
        
        Args:
            K: Number of fluorophores
            rng: Random number generator (unused, for compatibility)
            
        Returns:
            Array of shape (K, H*W) with rasterized abundance maps
        """
        from .spatial import FieldSpec
        
        H, W = self.field_shape
        A_maps = np.zeros((K, H, W), dtype=np.float32)
        
        # Helper functions for rendering (simplified versions)
        def add_circle(map_ref: Array, cy: int, cx: int, radius: float, amplitude: float) -> None:
            """Add filled circle to map."""
            int_radius = int(np.ceil(radius))
            y_min = max(0, cy - int_radius)
            y_max = min(H, cy + int_radius + 1)
            x_min = max(0, cx - int_radius)
            x_max = min(W, cx + int_radius + 1)
            
            if y_min >= y_max or x_min >= x_max:
                return
            
            y_local = np.arange(y_min, y_max)
            x_local = np.arange(x_min, x_max)
            yy_local, xx_local = np.meshgrid(y_local, x_local, indexing="ij")
            
            rr2 = (yy_local - cy) ** 2 + (xx_local - cx) ** 2
            circle_mask = rr2 <= (radius ** 2)
            map_ref[y_min:y_max, x_min:x_max][circle_mask] += amplitude
        
        def add_box(map_ref: Array, cy: int, cx: int, half: int, amplitude: float) -> None:
            """Add filled box to map."""
            y0 = max(0, cy - half)
            y1 = min(H, cy + half + 1)
            x0 = max(0, cx - half)
            x1 = min(W, cx + half + 1)
            map_ref[y0:y1, x0:x1] += amplitude
        
        def add_gaussian(map_ref: Array, cy: int, cx: int, sigma: float, amplitude: float) -> None:
            """Add Gaussian blob to map."""
            radius = int(np.ceil(4 * sigma))
            y_min = max(0, cy - radius)
            y_max = min(H, cy + radius + 1)
            x_min = max(0, cx - radius)
            x_max = min(W, cx + radius + 1)
            
            if y_min >= y_max or x_min >= x_max:
                return
            
            y_local = np.arange(y_min, y_max)
            x_local = np.arange(x_min, x_max)
            yy_local, xx_local = np.meshgrid(y_local, x_local, indexing="ij")
            
            g = np.exp(-((yy_local - cy) ** 2 + (xx_local - cx) ** 2) / (2.0 * sigma ** 2))
            map_ref[y_min:y_max, x_min:x_max] += amplitude * g
        
        # Render each object
        for obj in self.objects:
            cy = int(obj.position[0])
            cx = int(obj.position[1])
            
            for comp in obj.composition:
                k = comp['fluor_index']
                if k < 0 or k >= K:
                    continue
                
                amp = comp['intensity']
                
                if obj.type == 'circles':
                    add_circle(A_maps[k], cy, cx, obj.radius, amp)
                elif obj.type == 'boxes':
                    half = int(max(1, round(obj.radius)))
                    add_box(A_maps[k], cy, cx, half, amp)
                elif obj.type in ('gaussian_blobs', 'dots'):
                    add_gaussian(A_maps[k], cy, cx, obj.spot_sigma, amp)
        
        return A_maps.reshape(K, H * W)
    
    def __len__(self) -> int:
        """Return number of objects in scene."""
        return len(self.objects)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"GeometricScene({len(self.objects)} objects, field={self.field_shape})"

