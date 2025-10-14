from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

from .geometry import GeometricObject, GeometricScene


Array = np.ndarray


@dataclass
class FieldSpec:
    shape: Tuple[int, int]  # (H, W)
    pixel_size_nm: float


class AbundanceField:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def sample(self, K: int, field: FieldSpec, kind: str = "dots", **kwargs) -> Array:
        """Return A of shape (K, P).

        kinds:
          - 'dots': sparse Gaussian spots per fluor
          - 'uniform': uniform random non-negative
          - 'circles': filled disks per fluor
          - 'boxes': filled squares per fluor
          - 'gaussian_blobs': Gaussian blobs with configurable sigma
          - 'mixed': mixture of circles/boxes/gaussian_blobs per fluor
        """
        H, W = field.shape
        P = H * W
        if kind == "uniform":
            A = self.rng.random((K, P), dtype=float)
            return A.astype(np.float32)
        
        # Initialize output maps
        A_maps = np.zeros((K, H, W), dtype=np.float32)

        if kind == "dots":
            density = float(kwargs.get("density_per_100x100_um2", 50.0))
            spot_profile = kwargs.get("spot_profile", {"kind": "gaussian", "sigma_px": 1.2})
            sigma_px = float(spot_profile.get("sigma_px", 1.2))

            # Expected number of spots per 100x100 um^2
            pixel_size_um = field.pixel_size_nm / 1000.0
            area_um2 = (H * pixel_size_um) * (W * pixel_size_um)
            expected_spots = density * (area_um2 / 1.0e4)

            def add_local_gaussian(map_ref: Array, cy: int, cx: int, sigma: float, amplitude: float) -> None:
                """Add Gaussian using local computation for efficiency and accuracy."""
                # Calculate local region (4 sigma is enough for >99.9% of the Gaussian)
                radius = int(np.ceil(4 * sigma))
                
                # Bounds checking
                y_min = max(0, cy - radius)
                y_max = min(H, cy + radius + 1)
                x_min = max(0, cx - radius)
                x_max = min(W, cx + radius + 1)
                
                if y_min >= y_max or x_min >= x_max:
                    return  # Outside bounds
                
                # Local meshgrid only for this region
                y_local = np.arange(y_min, y_max)
                x_local = np.arange(x_min, x_max)
                yy_local, xx_local = np.meshgrid(y_local, x_local, indexing="ij")
                
                # Compute Gaussian only in local region
                g = np.exp(-((yy_local - cy) ** 2 + (xx_local - cx) ** 2) / (2.0 * sigma ** 2))
                
                # Add only to local region
                map_ref[y_min:y_max, x_min:x_max] += (amplitude * g).astype(np.float32)

            for k in range(K):
                n_spots = self.rng.poisson(lam=max(expected_spots, 0.0))
                if n_spots == 0:
                    continue
                for _ in range(int(n_spots)):
                    cy = int(self.rng.integers(0, H))
                    cx = int(self.rng.integers(0, W))
                    amp = float(self.rng.random()) + 0.5  # avoid too small amplitudes
                    add_local_gaussian(A_maps[k], cy, cx, sigma_px, amp)
            return A_maps.reshape(K, P)

        # New object-based generators
        count_per_fluor = int(kwargs.get("count_per_fluor", 50))
        size_px = float(kwargs.get("size_px", 6.0))
        intensity_min = float(kwargs.get("intensity_min", 0.5))
        intensity_max = float(kwargs.get("intensity_max", 1.5))

        def add_circle(map_ref: Array, cy: int, cx: int, radius: float, amplitude: float) -> None:
            """Add circle using local computation for efficiency."""
            # Calculate local region
            int_radius = int(np.ceil(radius))
            
            # Bounds checking
            y_min = max(0, cy - int_radius)
            y_max = min(H, cy + int_radius + 1)
            x_min = max(0, cx - int_radius)
            x_max = min(W, cx + int_radius + 1)
            
            if y_min >= y_max or x_min >= x_max:
                return  # Outside bounds
            
            # Local meshgrid only for this region
            y_local = np.arange(y_min, y_max)
            x_local = np.arange(x_min, x_max)
            yy_local, xx_local = np.meshgrid(y_local, x_local, indexing="ij")
            
            # Compute circle only in local region
            rr2 = (yy_local - cy) ** 2 + (xx_local - cx) ** 2
            circle_mask = rr2 <= (radius ** 2)
            
            # Add only to local region
            map_ref[y_min:y_max, x_min:x_max][circle_mask] += amplitude

        def add_box(map_ref: Array, cy: int, cx: int, half: int, amplitude: float) -> None:
            y0 = max(0, cy - half)
            y1 = min(H, cy + half + 1)
            x0 = max(0, cx - half)
            x1 = min(W, cx + half + 1)
            map_ref[y0:y1, x0:x1] += amplitude

        def add_gaussian(map_ref: Array, cy: int, cx: int, sigma: float, amplitude: float) -> None:
            """Add Gaussian using local computation for efficiency and accuracy."""
            # Calculate local region (4 sigma is enough for >99.9% of the Gaussian)
            radius = int(np.ceil(4 * sigma))
            
            # Bounds checking
            y_min = max(0, cy - radius)
            y_max = min(H, cy + radius + 1)
            x_min = max(0, cx - radius)
            x_max = min(W, cx + radius + 1)
            
            if y_min >= y_max or x_min >= x_max:
                return  # Outside bounds
            
            # Local meshgrid only for this region
            y_local = np.arange(y_min, y_max)
            x_local = np.arange(x_min, x_max)
            yy_local, xx_local = np.meshgrid(y_local, x_local, indexing="ij")
            
            # Compute Gaussian only in local region
            g = np.exp(-((yy_local - cy) ** 2 + (xx_local - cx) ** 2) / (2.0 * sigma ** 2))
            
            # Add only to local region
            map_ref[y_min:y_max, x_min:x_max] += amplitude * g

        def rand_amp() -> float:
            return float(self.rng.uniform(intensity_min, intensity_max))

        all_types = ("circles", "boxes", "gaussian_blobs")

        for k in range(K):
            n = count_per_fluor
            for _ in range(max(0, n)):
                cy = int(self.rng.integers(0, H))
                cx = int(self.rng.integers(0, W))
                amp = rand_amp()
                obj_type = kind
                if kind == "mixed":
                    obj_type = all_types[int(self.rng.integers(0, len(all_types)))]

                if obj_type == "circles":
                    add_circle(A_maps[k], cy, cx, radius=size_px, amplitude=amp)
                elif obj_type == "boxes":
                    add_box(A_maps[k], cy, cx, half=int(max(1, round(size_px))), amplitude=amp)
                elif obj_type == "gaussian_blobs":
                    add_gaussian(A_maps[k], cy, cx, sigma=size_px, amplitude=amp)
                else:
                    raise ValueError(f"Unknown object kind: {obj_type}")

        return A_maps.reshape(K, P)

    def build_from_objects(
        self,
        K: int,
        field: FieldSpec,
        objects: list,
        base: Array | None = None,
        track_objects: bool = True,
        use_ppp: bool = False,
    ) -> tuple[Array, GeometricScene]:
        """Build abundance maps from a list of object specs.

        Each object is a dict with keys:
          - 'fluor_index': int (0..K-1) OR 'composition': list of fluorophore compositions
          - 'kind': 'circles' | 'boxes' | 'gaussian_blobs' | 'dots'
          - 'region': {'type': 'full' | 'rect' | 'circle', ...}
          - 'count': int
          - 'size_px': float
          - 'intensity_min': float
          - 'intensity_max': float
          - 'spot_sigma': float (for 'dots'/'gaussian_blobs')
          - 'use_ppp': bool (optional, per-object PPP override)
          - 'radius_override': bool (optional, use explicit radius)
          - 'radius': float (optional, explicit radius value)
          
        Composition format (for multi-fluorophore objects):
          'composition': [
              {'fluor_index': 0, 'ratio': 0.6, 'ratio_noise': 0.1},
              {'fluor_index': 1, 'ratio': 0.4, 'ratio_noise': 0.05},
          ]
        
        Args:
            K: Number of fluorophores
            field: Field specification
            objects: List of object specifications
            base: Optional base abundance array
            track_objects: Whether to track geometric objects
            use_ppp: Whether to use Poisson Point Process for object counts
        
        Returns:
            tuple: (A_maps array of shape (K, P), GeometricScene with object geometry)
        """
        H, W = field.shape
        P = H * W
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

        A_maps = np.zeros((K, H, W), dtype=np.float32) if base is None else base.reshape(K, H, W).astype(np.float32)

        def region_mask(region: dict) -> Array:
            rtype = region.get("type", "full")
            if rtype == "full":
                return np.ones((H, W), dtype=bool)
            if rtype == "rect":
                x0 = int(max(0, region.get("x0", 0)))
                y0 = int(max(0, region.get("y0", 0)))
                w = int(max(1, region.get("w", W)))
                h = int(max(1, region.get("h", H)))
                x1 = min(W, x0 + w)
                y1 = min(H, y0 + h)
                mask = np.zeros((H, W), dtype=bool)
                mask[y0:y1, x0:x1] = True
                return mask
            if rtype == "circle":
                cx = float(region.get("cx", W / 2))
                cy = float(region.get("cy", H / 2))
                r = float(region.get("r", min(H, W) / 3))
                return ((yy - cy) ** 2 + (xx - cx) ** 2) <= (r ** 2)
            raise ValueError(f"Unknown region type: {rtype}")

        def add_circle(map_ref: Array, cy: int, cx: int, radius: float, amplitude: float, mask: Array) -> None:
            """Add circle using local computation for efficiency."""
            # Calculate local region
            int_radius = int(np.ceil(radius))
            
            # Bounds checking
            y_min = max(0, cy - int_radius)
            y_max = min(H, cy + int_radius + 1)
            x_min = max(0, cx - int_radius)
            x_max = min(W, cx + int_radius + 1)
            
            if y_min >= y_max or x_min >= x_max:
                return  # Outside bounds
            
            # Local meshgrid only for this region
            y_local = np.arange(y_min, y_max)
            x_local = np.arange(x_min, x_max)
            yy_local, xx_local = np.meshgrid(y_local, x_local, indexing="ij")
            
            # Compute circle only in local region
            rr2 = (yy_local - cy) ** 2 + (xx_local - cx) ** 2
            circle_mask = (rr2 <= (radius ** 2)).astype(np.float32)
            
            # Apply mask and add only to local region
            local_mask = mask[y_min:y_max, x_min:x_max]
            map_ref[y_min:y_max, x_min:x_max] += amplitude * circle_mask * local_mask

        def add_box(map_ref: Array, cy: int, cx: int, half: int, amplitude: float, mask: Array) -> None:
            y0 = max(0, cy - half)
            y1 = min(H, cy + half + 1)
            x0 = max(0, cx - half)
            x1 = min(W, cx + half + 1)
            local = np.zeros((H, W), dtype=np.float32)
            local[y0:y1, x0:x1] = amplitude
            map_ref += local * mask

        def add_gaussian(map_ref: Array, cy: int, cx: int, sigma: float, amplitude: float, mask: Array) -> None:
            """Add Gaussian using local computation for efficiency and accuracy."""
            # Calculate local region (4 sigma is enough for >99.9% of the Gaussian)
            radius = int(np.ceil(4 * sigma))
            
            # Bounds checking
            y_min = max(0, cy - radius)
            y_max = min(H, cy + radius + 1)
            x_min = max(0, cx - radius)
            x_max = min(W, cx + radius + 1)
            
            if y_min >= y_max or x_min >= x_max:
                return  # Outside bounds
            
            # Local meshgrid only for this region
            y_local = np.arange(y_min, y_max)
            x_local = np.arange(x_min, x_max)
            yy_local, xx_local = np.meshgrid(y_local, x_local, indexing="ij")
            
            # Compute Gaussian only in local region
            g = np.exp(-((yy_local - cy) ** 2 + (xx_local - cx) ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
            
            # Apply mask and add only to local region
            local_mask = mask[y_min:y_max, x_min:x_max]
            map_ref[y_min:y_max, x_min:x_max] += (amplitude * g) * local_mask

        generated_objects = []
        geometric_objects = []
        object_id_counter = 0
        
        def get_composition(obj: dict) -> list:
            """Extract fluorophore composition, handling both single and multi-fluorophore objects."""
            if 'composition' in obj:
                return obj['composition']
            else:
                # Legacy single fluorophore format
                return [{
                    'fluor_index': obj.get('fluor_index', 0),
                    'ratio': 1.0,
                    'ratio_noise': 0.0
                }]
        
        for spec_index, obj_spec in enumerate(objects or []):
            composition = get_composition(obj_spec)
            
            # Validate fluorophore indices
            valid_composition = [c for c in composition if 0 <= c['fluor_index'] < K]
            if not valid_composition:
                continue
                
            kind = obj_spec.get("kind", "gaussian_blobs")
            region = obj_spec.get("region", {"type": "full"})
            cnt = int(obj_spec.get("count", 50))
            size_px = float(obj_spec.get("size_px", 6.0))
            imin = float(obj_spec.get("intensity_min", 0.5))
            imax = float(obj_spec.get("intensity_max", 1.5))
            spot_sigma = float(obj_spec.get("spot_sigma", max(1.0, size_px / 3.0)))
            
            # Check for PPP sampling (per-object or global)
            obj_use_ppp = obj_spec.get('use_ppp', use_ppp)
            if obj_use_ppp:
                # Sample count from Poisson distribution
                area_px2 = H * W
                λ = cnt / area_px2
                cnt = int(self.rng.poisson(lam=λ * area_px2))

            mask = region_mask(region)

            # Generate individual objects
            for _ in range(max(0, cnt)):
                cy = int(self.rng.integers(0, H))
                cx = int(self.rng.integers(0, W))
                base_amp = float(self.rng.uniform(imin, imax))
                
                # Sample actual ratios with noise for this object
                actual_ratios = []
                for comp in valid_composition:
                    ratio = float(comp['ratio'])
                    noise = float(comp.get('ratio_noise', 0.0))
                    if noise > 0:
                        actual_ratio = max(0.0, float(self.rng.normal(ratio, noise)))
                    else:
                        actual_ratio = ratio
                    actual_ratios.append(actual_ratio)
                
                # Normalize ratios to sum to 1
                total = sum(actual_ratios)
                if total > 0:
                    actual_ratios = [r / total for r in actual_ratios]
                else:
                    actual_ratios = [1.0 / len(actual_ratios)] * len(actual_ratios)
                
                # Determine radius for geometric representation (always derived from object type)
                if kind == 'circles':
                    radius = size_px
                elif kind == 'boxes':
                    radius = size_px  # Treat as effective radius
                elif kind in ('gaussian_blobs', 'dots'):
                    radius = 2.0 * spot_sigma  # 95% containment radius
                else:
                    radius = 3.0  # Default fallback
                
                # Record object instance if tracking enabled
                if track_objects:
                    obj_instance = {
                        'id': object_id_counter,
                        'position': (float(cy), float(cx)),
                        'type': kind,
                        'base_intensity': float(base_amp),
                        'size_px': float(size_px),
                        'spot_sigma': float(spot_sigma),
                        'radius': float(radius),
                        'region': region.copy(),
                        'composition': []
                    }
                    
                    # Store actual composition with intensities
                    composition_list = []
                    for comp, actual_ratio in zip(valid_composition, actual_ratios):
                        comp_dict = {
                            'fluor_index': int(comp['fluor_index']),
                            'ratio': float(actual_ratio),
                            'intensity': float(base_amp * actual_ratio)
                        }
                        obj_instance['composition'].append(comp_dict)
                        composition_list.append(comp_dict)
                    
                    generated_objects.append(obj_instance)
                    
                    # Create GeometricObject for geometric scene
                    geometric_obj = GeometricObject(
                        id=object_id_counter,
                        position=(float(cy), float(cx)),
                        radius=float(radius),
                        composition=composition_list,
                        type=kind,
                        spot_sigma=float(spot_sigma),
                        base_intensity=float(base_amp),
                        size_px=float(size_px),
                        source_spec_index=spec_index
                    )
                    geometric_objects.append(geometric_obj)
                    
                    object_id_counter += 1
                
                # Add to abundance maps for each fluorophore in composition
                for comp, actual_ratio in zip(valid_composition, actual_ratios):
                    k = int(comp['fluor_index'])
                    amp = float(base_amp * actual_ratio)
                    
                    if kind == "circles":
                        add_circle(A_maps[k], cy, cx, radius=size_px, amplitude=amp, mask=mask)
                    elif kind == "boxes":
                        add_box(A_maps[k], cy, cx, half=int(max(1, round(size_px))), amplitude=amp, mask=mask)
                    elif kind in ("gaussian_blobs", "dots"):
                        add_gaussian(A_maps[k], cy, cx, sigma=spot_sigma, amplitude=amp, mask=mask)
                    else:
                        raise ValueError(f"Unknown object kind: {kind}")

        # Create geometric scene
        geometric_scene = GeometricScene(geometric_objects, field.shape)
        
        return A_maps.reshape(K, P), geometric_scene


class PSF:
    """Placeholder PSF for future extensions."""

    def __init__(self, sigma_px: float = 1.0):
        self.sigma_px = float(sigma_px)

    def kernel(self, field: FieldSpec) -> Array:
        H, W = field.shape
        size = int(6 * self.sigma_px + 1)
        size = max(3, size | 1)  # odd size
        c = size // 2
        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        g = np.exp(-(((yy - c) ** 2 + (xx - c) ** 2) / (2.0 * self.sigma_px ** 2)))
        g = g / np.sum(g)
        return g.astype(np.float32)


