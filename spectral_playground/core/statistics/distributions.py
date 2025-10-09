"""Radius distribution models for statistical analysis."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import truncnorm


Array = np.ndarray


class RadiusDistribution:
    """Manages radius distributions with moment computation.
    
    Currently supports truncated normal distribution, which provides a good
    approximation for typical fluorescent object size distributions while
    maintaining tractable closed-form moments.
    """
    
    def __init__(
        self,
        dist_type: str = 'truncated_normal',
        mean: float = 3.0,
        std: float = 0.6,
        r_min: float = 2.0,
        r_max: float = 5.0
    ):
        """Initialize radius distribution.
        
        Args:
            dist_type: Distribution type ('truncated_normal' only for now)
            mean: Mean radius (μ) in pixels
            std: Standard deviation (σ) in pixels
            r_min: Minimum radius in pixels
            r_max: Maximum radius in pixels
        """
        if dist_type != 'truncated_normal':
            raise ValueError(f"Unsupported distribution type: {dist_type}")
        
        self.dist_type = dist_type
        self.mean_param = mean
        self.std_param = std
        self.r_min = r_min
        self.r_max = r_max
        
        # Create scipy truncated normal distribution
        a = (r_min - mean) / std
        b = (r_max - mean) / std
        self._dist = truncnorm(a, b, loc=mean, scale=std)
    
    def sample(self, n: int, rng: np.random.Generator) -> Array:
        """Sample n radii from the distribution.
        
        Args:
            n: Number of samples
            rng: NumPy random generator
            
        Returns:
            Array of n sampled radii
        """
        # Use scipy's rvs with our rng for consistency
        return self._dist.rvs(size=n, random_state=rng)
    
    def pdf(self, r: float | Array) -> float | Array:
        """Probability density function.
        
        Args:
            r: Radius value(s)
            
        Returns:
            PDF value(s)
        """
        return self._dist.pdf(r)
    
    def cdf(self, r: float | Array) -> float | Array:
        """Cumulative distribution function.
        
        Args:
            r: Radius value(s)
            
        Returns:
            CDF value(s)
        """
        return self._dist.cdf(r)
    
    def moment(self, n: int) -> float:
        """Compute nth moment E[R^n].
        
        Args:
            n: Moment order
            
        Returns:
            E[R^n]
        """
        if n == 0:
            return 1.0
        
        # Use scipy's moment method
        return float(self._dist.moment(n))
    
    @property
    def mean(self) -> float:
        """Expected value E[R]."""
        return float(self._dist.mean())
    
    @property
    def std(self) -> float:
        """Standard deviation."""
        return float(self._dist.std())
    
    @property
    def support(self) -> Tuple[float, float]:
        """Support interval (r_min, r_max)."""
        return (self.r_min, self.r_max)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RadiusDistribution(type={self.dist_type}, "
            f"μ={self.mean_param:.2f}, σ={self.std_param:.2f}, "
            f"range=[{self.r_min:.2f}, {self.r_max:.2f}])"
        )

