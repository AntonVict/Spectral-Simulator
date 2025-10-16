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
    
    def __init__(self, objects: List[GeometricObject], field_shape: Tuple[int, int], 
                 overlap_mode: str = 'continuous'):
        """Initialize geometric scene.
        
        Args:
            objects: List of GeometricObject instances
            field_shape: (H, W) dimensions of the field in pixels
            overlap_mode: Overlap detection mode ('continuous' or 'pixelated')
        """
        self.objects = objects
        self.field_shape = field_shape
        self.overlap_mode = overlap_mode
        
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
            if self.overlap_mode == 'pixelated':
                self._precompute_overlaps_pixelated()
            else:  # continuous (default)
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
    
    def _precompute_overlaps_pixelated(self) -> None:
        """Precompute overlaps using pixel-based discretization.
        
        Rounds positions and radii to integers, then uses the same efficient
        distance-based overlap detection as continuous mode. This is much faster
        than pixel-by-pixel rasterization while still simulating discretization
        effects from real image analysis.
        """
        n = len(self.objects)
        self._overlap_graph = [[] for _ in range(n)]
        
        if n == 0:
            return
        
        # Performance guard for very large scenes
        if n > 200000:
            import warnings
            warnings.warn(
                f"Skipping pixelated overlap precomputation for {n} objects (would be very slow). "
                "Object-based crowding analysis will be unavailable.",
                UserWarning
            )
            return
        
        # Round all positions and radii to integers (simulate pixel discretization)
        centers_rounded = np.array([
            (round(obj.position[0]), round(obj.position[1])) 
            for obj in self.objects
        ])
        radii_rounded = np.array([
            max(1, round(obj.radius)) 
            for obj in self.objects
        ])
        max_radius = radii_rounded.max()
        
        # Build KD-tree on rounded centers for fast spatial queries
        tree_rounded = cKDTree(centers_rounded)
        
        # Check overlaps using rounded values (same algorithm as continuous mode)
        for i in range(n):
            # Query neighbors using rounded position
            search_radius = radii_rounded[i] + max_radius
            candidates = tree_rounded.query_ball_point(
                centers_rounded[i],
                r=search_radius
            )
            
            if len(candidates) == 0:
                continue
            
            # Vectorized distance computation using rounded centers
            candidate_centers = centers_rounded[candidates]
            dists = np.linalg.norm(
                candidate_centers - centers_rounded[i], 
                axis=1
            )
            candidate_radii = radii_rounded[candidates]
            
            # Check overlap condition with ROUNDED values
            overlaps = dists < (radii_rounded[i] + candidate_radii)
            
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
    
    @staticmethod
    def _circle_intersection_area(r1: float, r2: float, d: float) -> float:
        """Compute area of intersection between two circles.
        
        Args:
            r1: Radius of first circle
            r2: Radius of second circle
            d: Distance between centers
            
        Returns:
            Intersection area in square pixels
        """
        # No overlap
        if d >= r1 + r2:
            return 0.0
        
        # One circle fully inside the other
        if d <= abs(r1 - r2):
            return np.pi * min(r1, r2) ** 2
        
        # Partial overlap - use circular segment formula
        # Area = r1² * arccos((d² + r1² - r2²)/(2*d*r1)) 
        #      + r2² * arccos((d² + r2² - r1²)/(2*d*r2))
        #      - 0.5 * sqrt((r1+r2-d) * (d+r1-r2) * (d-r1+r2) * (d+r1+r2))
        
        d2 = d * d
        r1_2 = r1 * r1
        r2_2 = r2 * r2
        
        # Angle subtended by intersection in each circle
        alpha1 = np.arccos((d2 + r1_2 - r2_2) / (2 * d * r1))
        alpha2 = np.arccos((d2 + r2_2 - r1_2) / (2 * d * r2))
        
        # Area using circular segment formula
        area = (r1_2 * alpha1 + r2_2 * alpha2 - 
                0.5 * np.sqrt((r1 + r2 - d) * (d + r1 - r2) * 
                             (d - r1 + r2) * (d + r1 + r2)))
        
        return area
    
    def compute_overlap_metrics(self) -> Dict[str, Any]:
        """Compute overlap intensity metrics for pairs that DO overlap.
        
        Returns dict with:
            - overlap_pairs: [(i, j, overlap_depth, coverage_fraction, distance), ...]
            - per_object_max_depth: [max overlap_depth for each object]
            - per_object_max_coverage: [max coverage_fraction for each object]
        
        Metrics:
            - overlap_length = (r_i + r_j) - distance  [raw overlap distance in pixels]
            - overlap_depth = overlap_length / (r_i + r_j)  [0-1, symmetric]
                * 0 = just touching, 1 = centers coincide
                * For equal sizes: 0.5 = halfway overlapped
            - coverage_fraction = intersection_area / area_of_smaller_circle  [0-1]
                * Fraction of smaller circle's area covered by intersection
                * 1.0 = smaller circle fully covered/engulfed
        """
        n = len(self.objects)
        overlap_pairs = []
        per_object_max_depth = np.zeros(n)
        per_object_max_coverage = np.zeros(n)
        
        if n == 0:
            return {
                'overlap_pairs': overlap_pairs,
                'per_object_max_depth': per_object_max_depth,
                'per_object_max_coverage': per_object_max_coverage
            }
        
        # Get precomputed overlap graph
        overlap_graph = self.overlap_graph
        
        # Vectorized data
        centers = np.array([obj.position for obj in self.objects])
        radii = np.array([obj.radius for obj in self.objects])
        
        # Track per-object stats
        object_depths = [[] for _ in range(n)]
        object_coverages = [[] for _ in range(n)]
        
        # Compute metrics for each overlapping pair
        for i in range(n):
            neighbors = overlap_graph[i]
            if not neighbors:
                continue
            
            for j in neighbors:
                if i >= j:  # Only process each pair once
                    continue
                
                # Compute distance
                distance = np.linalg.norm(centers[i] - centers[j])
                r_i, r_j = radii[i], radii[j]
                r_sum = r_i + r_j
                
                # Compute overlap metrics
                overlap_length = r_sum - distance
                
                if overlap_length > 0:  # Should always be true for overlapping pairs
                    overlap_depth = overlap_length / r_sum
                    
                    # Compute area-based coverage fraction
                    intersection_area = self._circle_intersection_area(r_i, r_j, distance)
                    smaller_area = np.pi * min(r_i, r_j) ** 2
                    coverage_fraction = intersection_area / smaller_area
                    
                    # Store pair metrics
                    overlap_pairs.append((i, j, overlap_depth, coverage_fraction, distance))
                    
                    # Track for per-object stats
                    object_depths[i].append(overlap_depth)
                    object_depths[j].append(overlap_depth)
                    object_coverages[i].append(coverage_fraction)
                    object_coverages[j].append(coverage_fraction)
        
        # Compute per-object max values
        for i in range(n):
            if object_depths[i]:
                per_object_max_depth[i] = max(object_depths[i])
                per_object_max_coverage[i] = max(object_coverages[i])
        
        return {
            'overlap_pairs': overlap_pairs,
            'per_object_max_depth': per_object_max_depth,
            'per_object_max_coverage': per_object_max_coverage
        }
    
    def compute_proximity_metrics(self, epsilon: float) -> Dict[str, Any]:
        """Compute proximity metrics for near-miss objects within gap < epsilon.
        
        Args:
            epsilon: Max gap distance to consider (pixels)
        
        Returns dict with:
            - proximity_pairs: [(i, j, gap_distance, normalized_gap), ...]
            - per_object_near_count: [num near-neighbors per object]
        
        Metrics:
            - gap_distance = distance - (r_i + r_j)  [0 to epsilon]
            - normalized_gap = gap / min(r_i, r_j)  [size-independent]
        """
        n = len(self.objects)
        proximity_pairs = []
        per_object_near_count = np.zeros(n, dtype=int)
        
        if n == 0 or epsilon <= 0:
            return {
                'proximity_pairs': proximity_pairs,
                'per_object_near_count': per_object_near_count
            }
        
        # Vectorized data
        centers = np.array([obj.position for obj in self.objects])
        radii = np.array([obj.radius for obj in self.objects])
        max_radius = radii.max()
        
        # Use KD-tree for efficient spatial queries
        tree = self.spatial_index
        
        for i in range(n):
            # Query neighbors within r_i + max_radius + epsilon
            search_radius = radii[i] + max_radius + epsilon
            candidates = tree.query_ball_point(centers[i], r=search_radius)
            
            if not candidates:
                continue
            
            for j in candidates:
                if i >= j:  # Only process each pair once
                    continue
                
                # Compute distance and gap
                distance = np.linalg.norm(centers[i] - centers[j])
                r_sum = radii[i] + radii[j]
                gap_distance = distance - r_sum
                
                # Check if near-miss (positive gap within epsilon)
                if 0 < gap_distance < epsilon:
                    normalized_gap = gap_distance / min(radii[i], radii[j])
                    proximity_pairs.append((i, j, gap_distance, normalized_gap))
                    
                    # Increment counts
                    per_object_near_count[i] += 1
                    per_object_near_count[j] += 1
        
        return {
            'proximity_pairs': proximity_pairs,
            'per_object_near_count': per_object_near_count
        }
    
    def compute_epsilon_neighbor_curves(self, epsilon_values: List[float]) -> Dict[str, Any]:
        """Compute neighbor counts at multiple epsilon thresholds (stringency analysis).
        
        Args:
            epsilon_values: List of gap thresholds to test
        
        Returns dict with:
            - epsilon_vals: The input epsilon values
            - mean_neighbors: [mean neighbor count at each epsilon]
            - median_neighbors: [median at each epsilon]
            - per_object_counts: [[counts per object] for each epsilon]
        """
        n = len(self.objects)
        
        if n == 0 or not epsilon_values:
            return {
                'epsilon_vals': epsilon_values,
                'mean_neighbors': [],
                'median_neighbors': [],
                'per_object_counts': []
            }
        
        # Vectorized data
        centers = np.array([obj.position for obj in self.objects])
        radii = np.array([obj.radius for obj in self.objects])
        max_radius = radii.max()
        max_epsilon = max(epsilon_values)
        
        # Use KD-tree for efficient spatial queries
        tree = self.spatial_index
        
        # Storage for results
        mean_neighbors = []
        median_neighbors = []
        per_object_counts = []
        
        for epsilon in epsilon_values:
            counts = np.zeros(n, dtype=int)
            
            for i in range(n):
                # Query neighbors within r_i + max_radius + epsilon
                search_radius = radii[i] + max_radius + epsilon
                candidates = tree.query_ball_point(centers[i], r=search_radius)
                
                if not candidates:
                    continue
                
                for j in candidates:
                    if i == j:  # Skip self
                        continue
                    
                    # Compute distance and gap
                    distance = np.linalg.norm(centers[i] - centers[j])
                    r_sum = radii[i] + radii[j]
                    gap_distance = distance - r_sum
                    
                    # Count if within epsilon threshold
                    # Include both overlapping (gap < 0) and near-miss (0 <= gap < epsilon)
                    if gap_distance < epsilon:
                        counts[i] += 1
            
            # Compute statistics for this epsilon
            mean_neighbors.append(np.mean(counts))
            median_neighbors.append(np.median(counts))
            per_object_counts.append(counts.copy())
        
        return {
            'epsilon_vals': epsilon_values,
            'mean_neighbors': mean_neighbors,
            'median_neighbors': median_neighbors,
            'per_object_counts': per_object_counts
        }
    
    def compute_epsilon_margin_analysis(
        self,
        epsilon_values_empirical: List[float],
        epsilon_values_theoretical: List[float]
    ) -> Dict[str, Any]:
        """Analyze how many objects are naturally isolated with epsilon-margin requirement.
        
        Computes both empirical (isolation counting) and theoretical (Palm distribution)
        predictions for how many objects are epsilon-isolated (all neighbors have gap >= epsilon).
        
        Args:
            epsilon_values_empirical: Key epsilon values for empirical computation (few points)
            epsilon_values_theoretical: Epsilon values for theoretical curve (many points)
        
        Returns dict with:
            - epsilon_empirical: Epsilon values tested empirically
            - kept_empirical: Actual objects that are epsilon-isolated at each epsilon
            - kept_empirical_pct: Percentage isolated empirically
            - epsilon_theoretical: Epsilon values for theory curve
            - kept_theoretical: Predicted objects isolated (theory)
            - kept_theoretical_pct: Percentage isolated (theory)
            - survival_prob_theoretical: P(isolated) at each epsilon
        """
        n = len(self.objects)
        
        if n == 0:
            return {
                'epsilon_empirical': [],
                'kept_empirical': [],
                'kept_empirical_pct': [],
                'epsilon_theoretical': [],
                'kept_theoretical': [],
                'kept_theoretical_pct': [],
                'survival_prob_theoretical': []
            }
        
        # Compute theoretical predictions (fast!)
        H, W = self.field_shape
        area = H * W
        lambda_density = n / area
        
        radii = np.array([obj.radius for obj in self.objects])
        mean_radius = np.mean(radii)
        
        kept_theoretical = []
        kept_theoretical_pct = []
        survival_prob = []
        
        for eps in epsilon_values_theoretical:
            # Effective exclusion radius with epsilon margin
            # Need both object radii (central + neighbor) plus epsilon gap
            r_eff = 2 * mean_radius + eps
            exclusion_area = np.pi * r_eff ** 2
            
            # Palm distribution survival probability
            p_survive = np.exp(-lambda_density * exclusion_area)
            
            # Expected kept objects
            n_kept = n * p_survive
            pct_kept = 100 * p_survive
            
            kept_theoretical.append(n_kept)
            kept_theoretical_pct.append(pct_kept)
            survival_prob.append(p_survive)
        
        # Compute empirical values using isolation counting (slower)
        kept_empirical = []
        kept_empirical_pct = []
        
        for eps in epsilon_values_empirical:
            n_isolated_emp = self._count_epsilon_isolated(eps)
            pct_isolated_emp = 100 * n_isolated_emp / n if n > 0 else 0
            
            kept_empirical.append(n_isolated_emp)
            kept_empirical_pct.append(pct_isolated_emp)
        
        return {
            'epsilon_empirical': epsilon_values_empirical,
            'kept_empirical': kept_empirical,
            'kept_empirical_pct': kept_empirical_pct,
            'epsilon_theoretical': epsilon_values_theoretical,
            'kept_theoretical': kept_theoretical,
            'kept_theoretical_pct': kept_theoretical_pct,
            'survival_prob_theoretical': survival_prob
        }
    
    def _count_epsilon_isolated(self, epsilon: float) -> int:
        """Count objects that are naturally epsilon-isolated.
        
        An object is "epsilon-isolated" if ALL its neighbors have gap >= epsilon.
        This is similar to Overview's "isolated objects" but with stricter distance requirement.
        
        This is NOT optimization/greedy selection - just counting objects that satisfy the criterion.
        
        Args:
            epsilon: Required gap distance between object edges (pixels)
        
        Returns:
            Number of objects that are naturally epsilon-isolated
        """
        n = len(self.objects)
        if n == 0:
            return 0
        
        centers = np.array([obj.position for obj in self.objects])
        radii = np.array([obj.radius for obj in self.objects])
        max_radius = radii.max()
        tree = self.spatial_index
        
        isolated_count = 0
        
        for i in range(n):
            # Find all nearby objects (potential violators)
            search_radius = radii[i] + max_radius + epsilon
            candidates = tree.query_ball_point(centers[i], r=search_radius)
            
            # Check if this object is epsilon-isolated
            is_isolated = True
            for j in candidates:
                if i == j:
                    continue
                
                distance = np.linalg.norm(centers[i] - centers[j])
                gap = distance - (radii[i] + radii[j])
                
                if gap < epsilon:  # Has a neighbor too close!
                    is_isolated = False
                    break
            
            if is_isolated:
                isolated_count += 1
        
        return isolated_count
    
    def __len__(self) -> int:
        """Return number of objects in scene."""
        return len(self.objects)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"GeometricScene({len(self.objects)} objects, field={self.field_shape})"

