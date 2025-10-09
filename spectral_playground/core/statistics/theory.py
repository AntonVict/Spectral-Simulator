"""Theoretical predictions for Boolean model and point process statistics."""

from __future__ import annotations

from math import factorial
from typing import Optional

import numpy as np
from scipy.integrate import quad

from .distributions import RadiusDistribution


class BooleanModelTheory:
    """Theoretical predictions for Poisson Point Process with Boolean disc model.
    
    This class implements closed-form and semi-analytical formulas for:
    - Coverage probability
    - Box crowding probabilities
    - Palm distribution (object-centric overlaps)
    - Expected good target counts under various discard policies
    """
    
    def __init__(self, λ: float, R_dist: RadiusDistribution):
        """Initialize Boolean model theory.
        
        Args:
            λ: Intensity (objects per pixel²)
            R_dist: Radius distribution
        """
        self.λ = λ
        self.R_dist = R_dist
        
        # Precompute moments for efficiency
        self._E_R = R_dist.moment(1)
        self._E_R2 = R_dist.moment(2)
    
    def coverage_probability(self) -> float:
        """Compute theoretical coverage fraction.
        
        For a Boolean model with intensity λ and disc radii R,
        the coverage probability is:
            p = 1 - exp(-λπE[R²])
        
        Returns:
            Coverage probability (fraction of area covered)
        """
        return 1.0 - np.exp(-self.λ * np.pi * self._E_R2)
    
    def box_crowding_prob(
        self,
        a: float,
        k0: int,
        mode: str = 'germ'
    ) -> float:
        """Compute probability that a box is crowded.
        
        Args:
            a: Box side length (pixels)
            k0: Occupancy threshold (crowded if count >= k0)
            mode: Counting mode ('germ' or 'intersection')
        
        Returns:
            P(N_box >= k0) where N_box is object count in box
        """
        if mode == 'germ':
            # Germ-count mode: only centers inside box
            μ = self.λ * (a ** 2)
        elif mode == 'intersection':
            # Intersection mode: disc intersects box (Minkowski dilation)
            # Area = a² + 4aE[R] + πE[R²]
            μ = self.λ * (a**2 + 4*a*self._E_R + np.pi*self._E_R2)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Poisson tail probability: P(N >= k0) = 1 - P(N < k0)
        prob_less = sum(
            np.exp(-μ) * (μ ** k) / factorial(k)
            for k in range(k0)
        )
        
        return 1.0 - prob_less
    
    def expected_crowded_boxes(
        self,
        field_shape: tuple[int, int],
        a: int,
        k0: int,
        mode: str = 'germ'
    ) -> float:
        """Compute expected number of crowded boxes in a grid.
        
        Args:
            field_shape: (H, W) field dimensions
            a: Box side length
            k0: Occupancy threshold
            mode: Counting mode
        
        Returns:
            Expected number of crowded boxes
        """
        H, W = field_shape
        M = (H // a) * (W // a)  # Number of boxes
        p_crowded = self.box_crowding_prob(a, k0, mode)
        
        return M * p_crowded
    
    def palm_survival_probability(self, m: int) -> float:
        """Compute survival probability under object-based discard policy.
        
        Under the Palm distribution (conditioning on a typical object),
        the number of overlapping neighbors D follows a Poisson distribution
        with parameter ν(R) = λπ(R² + 2RE[R] + E[R²]).
        
        An object survives if D < m.
        
        Args:
            m: Neighbor threshold (discard if >= m neighbors)
        
        Returns:
            P(survive) = E[P(D < m | R)]
        """
        def survival_given_r(r: float) -> float:
            """P(D < m | R = r)."""
            # Palm intensity: objects within distance R + R' of center
            ν = self.λ * np.pi * (r**2 + 2*r*self._E_R + self._E_R2)
            
            # Poisson probability: P(D < m) = sum_{k=0}^{m-1} e^{-ν} ν^k / k!
            prob = sum(
                np.exp(-ν) * (ν ** k) / factorial(k)
                for k in range(m)
            )
            
            return prob
        
        def integrand(r: float) -> float:
            """Integrand: survival(r) * pdf(r)."""
            return survival_given_r(r) * self.R_dist.pdf(r)
        
        # Integrate over radius distribution
        result, error = quad(
            integrand,
            self.R_dist.support[0],
            self.R_dist.support[1],
            limit=100  # Increase subdivisions for accuracy
        )
        
        return result
    
    def palm_survival_probability_approx(self, m: int) -> float:
        """Approximate survival probability using mean radius.
        
        This is faster than full integration and accurate when R has low variance.
        
        Args:
            m: Neighbor threshold
        
        Returns:
            Approximate P(survive)
        """
        # Use mean ν computed at mean radius
        ν_mean = self.λ * np.pi * (
            self._E_R**2 + 2*self._E_R*self._E_R + self._E_R2
        )
        
        prob = sum(
            np.exp(-ν_mean) * (ν_mean ** k) / factorial(k)
            for k in range(m)
        )
        
        return prob
    
    def expected_good_count(
        self,
        n: int,
        m: int,
        use_approx: bool = False
    ) -> float:
        """Compute expected number of good targets under object policy.
        
        Args:
            n: Total number of objects
            m: Neighbor threshold
            use_approx: Use fast approximation instead of integration
        
        Returns:
            Expected number of objects surviving object-based policy
        """
        if use_approx:
            p_survive = self.palm_survival_probability_approx(m)
        else:
            p_survive = self.palm_survival_probability(m)
        
        return n * p_survive
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BooleanModelTheory(λ={self.λ:.4f}, "
            f"E[R]={self._E_R:.2f}, E[R²]={self._E_R2:.2f})"
        )

