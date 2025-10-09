"""Empirical crowding analysis using geometric scene data."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple, Any

import numpy as np

from spectral_playground.core.geometry import GeometricScene


Array = np.ndarray


class CrowdingAnalyzer:
    """Analyzer for empirical crowding metrics on geometric scenes.
    
    This class computes crowding statistics directly from geometric object
    representations without requiring rasterization, enabling fast analysis
    of large scenes.
    """
    
    def __init__(self, scene: GeometricScene):
        """Initialize analyzer with geometric scene.
        
        Args:
            scene: GeometricScene containing objects and spatial index
        """
        self.scene = scene
        self.H, self.W = scene.field_shape
        
        # Cache max radius for spatial queries (computed once)
        self._max_radius = max((obj.radius for obj in scene.objects), default=0.0)
    
    def analyze_box_crowding(
        self,
        a: int,
        k0: int,
        mode: str = 'germ'
    ) -> Dict[str, Any]:
        """Analyze box-based crowding.
        
        Divides field into non-overlapping boxes and identifies objects
        contributing to crowded boxes (occupancy >= k0).
        
        Args:
            a: Box side length (pixels)
            k0: Occupancy threshold
            mode: Counting mode ('germ' counts centers, 'intersection' checks overlap)
        
        Returns:
            Dictionary with:
                - total_objects: Total count
                - crowded_boxes: Number of boxes with occupancy >= k0
                - discarded_by_box: Number of objects in crowded boxes
                - good_targets: Objects not in any crowded box
                - good_per_area: Good targets per pixel²
                - crowded_box_positions: List of (i, j, count) for crowded boxes
                - discarded_object_ids: Set of discarded object IDs
        """
        n_objects = len(self.scene.objects)
        
        if n_objects == 0:
            return {
                'total_objects': 0,
                'crowded_boxes': 0,
                'discarded_by_box': 0,
                'good_targets': 0,
                'good_per_area': 0.0,
                'crowded_box_positions': [],
                'discarded_object_ids': set(),
            }
        
        # Create grid of boxes
        grid_H = self.H // a
        grid_W = self.W // a
        
        # Count objects in each box
        crowded_boxes = []
        box_to_objects: Dict[Tuple[int, int], List[int]] = {}
        
        for i in range(grid_H):
            for j in range(grid_W):
                box_y0 = i * a
                box_x0 = j * a
                box_y1 = box_y0 + a
                box_x1 = box_x0 + a
                
                box_center = (box_y0 + a/2, box_x0 + a/2)
                
                # Find objects in/intersecting this box
                if mode == 'germ':
                    # Count centers inside box
                    obj_ids = self._find_centers_in_box(
                        box_y0, box_x0, box_y1, box_x1
                    )
                elif mode == 'intersection':
                    # Count objects whose disc intersects box
                    obj_ids = self._find_discs_intersecting_box(
                        box_y0, box_x0, box_y1, box_x1
                    )
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                
                count = len(obj_ids)
                
                if count >= k0:
                    crowded_boxes.append((i, j, count))
                    box_to_objects[(i, j)] = obj_ids
        
        # Collect all objects in any crowded box
        discarded_ids: Set[int] = set()
        for box_ids in box_to_objects.values():
            discarded_ids.update(box_ids)
        
        good_targets = n_objects - len(discarded_ids)
        area_px2 = self.H * self.W
        
        return {
            'total_objects': n_objects,
            'crowded_boxes': len(crowded_boxes),
            'discarded_by_box': len(discarded_ids),
            'good_targets': good_targets,
            'good_per_area': good_targets / area_px2,
            'crowded_box_positions': crowded_boxes,
            'discarded_object_ids': discarded_ids,
        }
    
    def _find_centers_in_box(
        self,
        y0: float,
        x0: float,
        y1: float,
        x1: float
    ) -> List[int]:
        """Find object IDs with centers inside box (optimized with spatial index)."""
        # Query spatial index for candidates (uses KD-tree for O(log n) per query)
        box_cy = (y0 + y1) / 2
        box_cx = (x0 + x1) / 2
        search_radius = np.sqrt((y1 - y0)**2 + (x1 - x0)**2) / 2  # Half-diagonal
        
        candidate_indices = self.scene.spatial_index.query_ball_point(
            [box_cy, box_cx], 
            r=search_radius
        )
        
        # Filter to actual box bounds
        obj_ids = []
        for idx in candidate_indices:
            obj = self.scene.objects[idx]
            cy, cx = obj.position
            if y0 <= cy < y1 and x0 <= cx < x1:
                obj_ids.append(obj.id)
        return obj_ids
    
    def _find_discs_intersecting_box(
        self,
        y0: float,
        x0: float,
        y1: float,
        x1: float
    ) -> List[int]:
        """Find object IDs whose disc intersects box (optimized with spatial index).
        
        A disc intersects the box if the center is within distance R
        from the box (using Minkowski sum logic).
        """
        # Box center and dimensions
        box_cy = (y0 + y1) / 2
        box_cx = (x0 + x1) / 2
        half_h = (y1 - y0) / 2
        half_w = (x1 - x0) / 2
        
        # Search radius: box half-diagonal + max object radius (cached)
        box_half_diag = np.sqrt(half_h**2 + half_w**2)
        search_radius = box_half_diag + self._max_radius
        
        # Query spatial index for candidates
        candidate_indices = self.scene.spatial_index.query_ball_point(
            [box_cy, box_cx], 
            r=search_radius
        )
        
        # Filter to actual intersections
        obj_ids = []
        for idx in candidate_indices:
            obj = self.scene.objects[idx]
            cy, cx = obj.position
            
            # Distance from disc center to box (using clamping)
            closest_y = np.clip(cy, y0, y1)
            closest_x = np.clip(cx, x0, x1)
            
            dist = np.sqrt((cy - closest_y)**2 + (cx - closest_x)**2)
            
            # Disc intersects if distance <= radius
            if dist <= obj.radius:
                obj_ids.append(obj.id)
        
        return obj_ids
    
    def analyze_object_policy(self, m: int) -> Dict[str, Any]:
        """Analyze object-based overlap policy.
        
        Discards objects that have >= m geometric neighbors (overlapping discs).
        Uses precomputed overlap graph from scene.
        
        Args:
            m: Neighbor threshold
        
        Returns:
            Dictionary with:
                - total_objects: Total count
                - discarded_by_object: Number with >= m neighbors
                - good_targets: Objects with < m neighbors
                - good_per_area: Good targets per pixel²
                - discarded_object_ids: Set of discarded IDs
                - neighbor_counts: List of neighbor counts for each object
        """
        n_objects = len(self.scene.objects)
        
        if n_objects == 0:
            return {
                'total_objects': 0,
                'discarded_by_object': 0,
                'good_targets': 0,
                'good_per_area': 0.0,
                'discarded_object_ids': set(),
                'neighbor_counts': [],
            }
        
        # Count neighbors for each object using precomputed overlap graph
        neighbor_counts = []
        discarded_ids: Set[int] = set()
        
        for obj in self.scene.objects:
            n_neighbors = len(self.scene.get_neighbors(obj.id))
            neighbor_counts.append(n_neighbors)
            
            if n_neighbors >= m:
                discarded_ids.add(obj.id)
        
        good_targets = n_objects - len(discarded_ids)
        area_px2 = self.H * self.W
        
        return {
            'total_objects': n_objects,
            'discarded_by_object': len(discarded_ids),
            'good_targets': good_targets,
            'good_per_area': good_targets / area_px2,
            'discarded_object_ids': discarded_ids,
            'neighbor_counts': neighbor_counts,
        }
    
    def compute_coverage_fraction(self) -> float:
        """Compute empirical coverage fraction.
        
        Rasterizes all discs as boolean masks and computes fraction of
        covered pixels.
        
        Returns:
            Coverage fraction (0 to 1)
        """
        if len(self.scene.objects) == 0:
            return 0.0
        
        # Create boolean mask
        coverage_mask = np.zeros((self.H, self.W), dtype=bool)
        
        for obj in self.scene.objects:
            cy, cx = obj.position
            radius = obj.radius
            
            # Rasterize disc
            y_min = max(0, int(cy - radius))
            y_max = min(self.H, int(cy + radius + 1))
            x_min = max(0, int(cx - radius))
            x_max = min(self.W, int(cx + radius + 1))
            
            if y_min >= y_max or x_min >= x_max:
                continue
            
            # Create local grid
            y_coords = np.arange(y_min, y_max)
            x_coords = np.arange(x_min, x_max)
            yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
            
            # Check which pixels are inside disc
            dist_sq = (yy - cy)**2 + (xx - cx)**2
            inside = dist_sq <= radius**2
            
            # Update mask
            coverage_mask[y_min:y_max, x_min:x_max] |= inside
        
        return float(np.mean(coverage_mask))
    
    def compute_coverage_fraction_monte_carlo(self, n_samples: int = 10000) -> float:
        """Compute coverage fraction using Monte Carlo sampling with spatial index.
        
        Instead of rasterizing the entire field, randomly sample points and
        check if they're inside any object disc. Uses spatial index to only
        check nearby objects for each sample point.
        
        Args:
            n_samples: Number of random points to sample (default: 10,000)
        
        Returns:
            Estimated coverage fraction (0 to 1)
        """
        if len(self.scene.objects) == 0:
            return 0.0
        
        # Generate random sample points uniformly across the field
        rng = np.random.default_rng()
        sample_y = rng.uniform(0, self.H, n_samples)
        sample_x = rng.uniform(0, self.W, n_samples)
        
        # For each sample point, use spatial index to find nearby objects
        covered_count = 0
        search_radius = self._max_radius  # Only need to check objects within max_radius
        
        for y, x in zip(sample_y, sample_x):
            # Query spatial index for nearby objects
            nearby_indices = self.scene.spatial_index.query_ball_point([y, x], r=search_radius)
            
            # Check if point is inside any nearby disc
            for idx in nearby_indices:
                obj = self.scene.objects[idx]
                cy, cx = obj.position
                dist_sq = (y - cy)**2 + (x - cx)**2
                if dist_sq <= obj.radius**2:
                    covered_count += 1
                    break  # Point is covered, no need to check more objects
        
        return float(covered_count / n_samples)
    
    def combined_analysis(
        self,
        a: int,
        k0: int,
        m: int,
        mode: str = 'germ',
        compute_coverage: bool = True,
        use_monte_carlo: bool = True
    ) -> Dict[str, Any]:
        """Run both box and object analyses and combine results.
        
        Args:
            a: Box side length
            k0: Box occupancy threshold
            m: Object neighbor threshold
            mode: Box counting mode
            compute_coverage: If True, compute coverage (default: True)
            use_monte_carlo: If True, use fast Monte Carlo approximation; 
                           if False, use exact rasterization (slow!) (default: True)
        
        Returns:
            Combined dictionary with all metrics plus:
                - discarded_either: Objects failing either policy
                - discarded_both: Objects failing both policies
                - good_targets_strict: Objects passing both policies
                - coverage_fraction: Estimated or exact coverage fraction
        """
        box_result = self.analyze_box_crowding(a, k0, mode)
        obj_result = self.analyze_object_policy(m)
        
        # Compute coverage (fast Monte Carlo by default, exact rasterization if requested)
        if compute_coverage:
            if use_monte_carlo:
                coverage = self.compute_coverage_fraction_monte_carlo()
            else:
                coverage = self.compute_coverage_fraction()
        else:
            coverage = 0.0  # Explicitly skipped
        
        # Compute overlap statistics
        box_discarded = box_result['discarded_object_ids']
        obj_discarded = obj_result['discarded_object_ids']
        
        discarded_either = box_discarded | obj_discarded
        discarded_both = box_discarded & obj_discarded
        good_strict = len(self.scene) - len(discarded_either)
        
        return {
            **box_result,
            'discarded_by_object': obj_result['discarded_by_object'],
            'neighbor_counts': obj_result['neighbor_counts'],
            'coverage_fraction': coverage,
            'discarded_either': len(discarded_either),
            'discarded_both': len(discarded_both),
            'good_targets_strict': good_strict,
            'good_per_area_strict': good_strict / (self.H * self.W),
        }

