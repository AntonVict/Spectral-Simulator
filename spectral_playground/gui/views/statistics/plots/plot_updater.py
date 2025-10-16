"""Plot update logic for statistics analysis visualizations."""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
from matplotlib.axes import Axes

from spectral_playground.core.statistics import BooleanModelTheory


class PlotUpdater:
    """Utility class for updating statistics plots."""
    
    @staticmethod
    def update_survival_vs_lambda(
        ax: Axes,
        theory: BooleanModelTheory,
        results: Dict[str, Any],
        λ_current: float,
        policy: str,
        params: Dict[str, Any]
    ) -> None:
        """Update survival probability vs intensity plot.
        
        Args:
            ax: Matplotlib axes to plot on
            theory: Boolean model theory instance
            results: Analysis results dictionary
            λ_current: Current intensity value
            policy: Active policy ('overlap' or 'box')
            params: Policy-specific parameters
        """
        λ_range = np.linspace(0.0001, λ_current * 2.5, 100)
        
        if policy == 'overlap':
            m = params['neighbor_threshold']
            survival_curve = []
            for λ in λ_range:
                theory_λ = BooleanModelTheory(λ=λ, R_dist=theory.R_dist)
                p_survive = theory_λ.palm_survival_probability(m)
                survival_curve.append(p_survive)
            
            # Calculate empirical survival
            p_survive_empirical = results.get('isolated_objects', 0) / max(1, results['total_objects'])
            
            ax.plot(λ_range, survival_curve, 'b-', label=f'Theory (m={m})', linewidth=2)
            ax.plot(
                [λ_current], [p_survive_empirical],
                'ro', markersize=10, label='Empirical'
            )
            ax.set_title(f'Survival Probability vs Density (m={m})')
        else:  # box
            a = params['box_size']
            k0 = params['box_threshold']
            mode = params['count_mode']
            
            survival_curve = []
            for λ in λ_range:
                theory_λ = BooleanModelTheory(λ=λ, R_dist=theory.R_dist)
                p_crowded = theory_λ.box_crowding_prob(a, k0, mode)
                survival_curve.append(1 - p_crowded)
            
            # Calculate empirical survival
            p_survive_empirical = results.get('isolated_objects', 0) / max(1, results['total_objects'])
            
            ax.plot(λ_range, survival_curve, 'b-', 
                   label=f'Theory (a={a}, k₀={k0})', linewidth=2)
            ax.plot(
                [λ_current], [p_survive_empirical],
                'ro', markersize=10, label='Empirical'
            )
            ax.set_title(f'Survival Probability vs Density (a={a}, k₀={k0})')
        
        ax.set_xlabel('Density λ (objects per px²)')
        ax.set_ylabel('P(survive)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    @staticmethod
    def update_survival_vs_threshold(
        ax: Axes,
        theory: BooleanModelTheory,
        results: Dict[str, Any],
        λ_current: float,
        policy: str,
        params: Dict[str, Any]
    ) -> None:
        """Update survival probability vs threshold parameter plot.
        
        Args:
            ax: Matplotlib axes to plot on
            theory: Boolean model theory instance
            results: Analysis results dictionary
            λ_current: Current density value
            policy: Active policy ('overlap' or 'box')
            params: Policy-specific parameters
        """
        if policy == 'overlap':
            # Vary neighbor threshold m
            m_range = np.arange(1, 21)
            survival_curve = []
            for m_val in m_range:
                p_survive = theory.palm_survival_probability(int(m_val))
                survival_curve.append(p_survive)
            
            m_current = params['neighbor_threshold']
            p_survive_empirical = results.get('isolated_objects', 0) / max(1, results['total_objects'])
            
            ax.plot(m_range, survival_curve, 'b-', 
                   label=f'Theory (density λ={λ_current:.6f})', linewidth=2)
            ax.plot(
                [m_current], [p_survive_empirical],
                'ro', markersize=10, label='Empirical'
            )
            ax.set_title(f'Survival Probability vs Neighbor Threshold (λ={λ_current:.6f})')
            ax.set_xlabel('Neighbor threshold (m)')
        else:  # box
            # Vary occupancy threshold k0
            a = params['box_size']
            mode = params['count_mode']
            k0_range = np.arange(1, 21)
            survival_curve = []
            for k0_val in k0_range:
                p_crowded = theory.box_crowding_prob(a, int(k0_val), mode)
                survival_curve.append(1 - p_crowded)
            
            k0_current = params['box_threshold']
            p_survive_empirical = results.get('isolated_objects', 0) / max(1, results['total_objects'])
            
            ax.plot(k0_range, survival_curve, 'b-',
                   label=f'Theory (λ={λ_current:.6f}, a={a})', linewidth=2)
            ax.plot(
                [k0_current], [p_survive_empirical],
                'ro', markersize=10, label='Empirical'
            )
            ax.set_title(f'Survival Probability vs Occupancy Threshold (λ={λ_current:.6f})')
            ax.set_xlabel('Occupancy threshold (k₀)')
        
        ax.set_ylabel('P(survive)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    @staticmethod
    def update_isolated_count(
        ax: Axes,
        theory: BooleanModelTheory,
        results: Dict[str, Any],
        area_px2: float,
        policy: str,
        params: Dict[str, Any]
    ) -> None:
        """Update isolated objects vs density plot.
        
        Args:
            ax: Matplotlib axes to plot on
            theory: Boolean model theory instance
            results: Analysis results dictionary
            area_px2: Total area in pixels squared
            policy: Active policy ('overlap' or 'box')
            params: Policy-specific parameters
        """
        n_range = np.linspace(10, max(20.0, results['total_objects'] * 1.5), 100)
        
        # Compute theory curve based on active policy
        expected_good_curve = []
        if policy == 'overlap':
            m = params['neighbor_threshold']
            for n in n_range:
                λ_n = float(n) / area_px2
                theory_n = BooleanModelTheory(λ=λ_n, R_dist=theory.R_dist)
                p_survive_n = theory_n.palm_survival_probability(m)
                expected_good_curve.append(n * p_survive_n)
            theory_label = f'Theory (Neighbor Policy, m={m})'
        else:  # box policy
            a = params['box_size']
            k0 = params['box_threshold']
            mode = params['count_mode']
            for n in n_range:
                λ_n = float(n) / area_px2
                theory_n = BooleanModelTheory(λ=λ_n, R_dist=theory.R_dist)
                p_crowded = theory_n.box_crowding_prob(a, k0, mode)
                expected_good_curve.append(n * (1 - p_crowded))
            theory_label = f'Theory (Region Policy, a={a}, k₀={k0})'
        
        ax.plot(
            n_range,
            expected_good_curve,
            'b-',
            label=theory_label,
            linewidth=2
        )
        ax.plot(
            [results['total_objects']],
            [results.get('isolated_objects', 0)],
            'ro',
            markersize=10,
            label='Empirical'
        )
        ax.set_title('Isolated Objects vs Density')
        ax.set_xlabel('Total Objects (n)')
        ax.set_ylabel('Isolated Objects')
        ax.legend()
        ax.grid(True, alpha=0.3)

