"""
Breakage Model
==============

Hammer-mill breakage: selection + breakage function driven by impact energy.

The BreakageModel applies impact comminution kinetics (selection probability
from impact energy and particle size; Gaudin–Schuhmann daughter distribution).
It is fed impact flags and impact energies from the impact solver (hammer–
particle collisions), so breakage is coupled to hammer milling physics.

The BreakageModel:
    - Manages breakage parameters (tuned for legume/pea milling)
    - Tracks size distribution evolution
    - Calls breakage kernel for Lagrangian particles
    - Provides population balance for PSD-based simulation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..kernels import (
    breakage_step_np,
    breakage_psd_np,
    generate_fragments_np,
    WARP_AVAILABLE,
)
if WARP_AVAILABLE:
    import warp as wp
    from ..kernels import breakage_step_warp

from ..config import BreakageParams


@dataclass
class BreakageStats:
    """Statistics from breakage events."""

    num_breakage_events: int = 0
    total_mass_broken: float = 0.0
    size_reduction_ratio: float = 1.0
    num_fragments_created: int = 0


@dataclass
class BreakageModel:
    """Model for particle breakage in the hammer mill.

    Supports both Lagrangian (particle-based) and Eulerian (PSD-based)
    approaches to breakage simulation.
    """

    # Breakage parameters
    params: BreakageParams = field(default_factory=BreakageParams)

    # Device
    device: str = "cpu"

    # Random state
    _rng: Optional[np.random.Generator] = None

    # Warp arrays (lazy init)
    _wp_rand_states: Optional["wp.array"] = None
    _wp_break_flags: Optional["wp.array"] = None

    # Statistics
    _stats: BreakageStats = field(default_factory=BreakageStats)

    def __post_init__(self):
        if self._rng is None:
            self._rng = np.random.default_rng()

    def step_lagrangian(
        self,
        sizes: np.ndarray,
        masses: np.ndarray,
        impact_flags: np.ndarray,
        impact_energies: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
               Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int]:
        """Apply breakage to Lagrangian particles with optional multi-fragment generation.

        Args:
            sizes: Particle sizes [n]
            masses: Particle masses [n]
            impact_flags: Which particles were impacted [n]
            impact_energies: Impact energies [n]

        Returns:
            (new_sizes, new_masses, break_flags,
             frag_sizes, frag_masses, parent_indices, num_fragments_created)
            When enable_multi_fragment is False, fragment arrays are None and count is 0.
        """
        p = self.params

        # Save originals before kernel (needed for mass-conserving fragmentation)
        original_sizes = sizes.copy()
        original_masses = masses.copy()

        # Run existing breakage kernel (produces primary daughters)
        if self.device == "cuda" and WARP_AVAILABLE:
            new_sizes, new_masses, break_flags = self._step_warp(
                sizes, masses, impact_flags, impact_energies,
            )
        else:
            new_sizes, new_masses, break_flags = self._step_np(
                sizes, masses, impact_flags, impact_energies,
            )

        # Multi-fragment post-processing
        frag_sizes = None
        frag_masses = None
        parent_indices = None
        num_frags = 0

        if p.enable_multi_fragment and int(break_flags.sum()) > 0:
            (
                new_sizes, new_masses,
                frag_sizes, frag_masses,
                parent_indices, num_frags,
            ) = generate_fragments_np(
                parent_sizes=original_sizes,
                parent_masses=original_masses,
                primary_sizes=new_sizes,
                primary_masses=new_masses,
                break_flags=break_flags,
                impact_energies=impact_energies,
                gamma=p.breakage_distribution_exponent,
                min_size=p.d_min_um * 1e-6,
                max_fragments=p.max_fragments_per_event,
                frag_count_coeff=p.fragment_count_coefficient,
                frag_count_size_exp=p.fragment_count_size_exp,
                frag_count_energy_exp=p.fragment_count_energy_exp,
                energy_ref=p.min_impact_energy_j,
                rng=self._rng,
                coarse_threshold=p.regime_coarse_threshold_m,
                fine_threshold=p.regime_fine_threshold_m,
                gamma_coarse=p.gamma_coarse,
                gamma_fine=p.gamma_fine,
            )
            self._stats.num_fragments_created = num_frags

        return (new_sizes, new_masses, break_flags,
                frag_sizes, frag_masses, parent_indices, num_frags)

    def _step_np(
        self,
        sizes: np.ndarray,
        masses: np.ndarray,
        impact_flags: np.ndarray,
        impact_energies: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """NumPy implementation."""
        p = self.params

        new_sizes, new_masses, break_flags = breakage_step_np(
            sizes=sizes,
            masses=masses,
            impact_flags=impact_flags,
            impact_energies=impact_energies,
            selection_k=p.selection_rate_constant,
            selection_alpha=p.selection_size_exponent,
            reference_size=p.selection_reference_size_um * 1e-6,
            min_energy=p.min_impact_energy_j,
            energy_scale=p.energy_to_breakage_factor,
            breakage_gamma=p.breakage_distribution_exponent,
            min_size=p.d_min_um * 1e-6,
            rng=self._rng,
            coarse_threshold=p.regime_coarse_threshold_m,
            fine_threshold=p.regime_fine_threshold_m,
            gamma_coarse=p.gamma_coarse,
            gamma_fine=p.gamma_fine,
            clamp_lo_coarse=p.clamp_lo_coarse,
            clamp_hi_coarse=p.clamp_hi_coarse,
            clamp_lo_medium=p.clamp_lo_medium,
            clamp_hi_medium=p.clamp_hi_medium,
            clamp_lo_fine=p.clamp_lo_fine,
            clamp_hi_fine=p.clamp_hi_fine,
        )

        self._update_stats(sizes, new_sizes, masses, break_flags)

        return new_sizes, new_masses, break_flags

    def _step_warp(
        self,
        sizes: np.ndarray,
        masses: np.ndarray,
        impact_flags: np.ndarray,
        impact_energies: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Warp GPU implementation."""
        n = len(sizes)
        p = self.params

        # Convert to Warp arrays
        wp_sizes = wp.array(sizes, dtype=float, device="cuda")
        wp_masses = wp.array(masses, dtype=float, device="cuda")
        wp_impact_flags = wp.array(impact_flags, dtype=int, device="cuda")
        wp_impact_energies = wp.array(impact_energies, dtype=float, device="cuda")

        # Initialize random states if needed
        if self._wp_rand_states is None or len(self._wp_rand_states) != n:
            rand_seeds = self._rng.integers(0, 2**31, size=n, dtype=np.uint32)
            self._wp_rand_states = wp.array(rand_seeds, dtype=wp.uint32, device="cuda")
            self._wp_break_flags = wp.zeros(n, dtype=int, device="cuda")

        # Launch kernel
        breakage_step_warp(
            sizes=wp_sizes,
            masses=wp_masses,
            impact_flags=wp_impact_flags,
            impact_energies=wp_impact_energies,
            break_flags=self._wp_break_flags,
            rand_states=self._wp_rand_states,
            selection_k=p.selection_rate_constant,
            selection_alpha=p.selection_size_exponent,
            reference_size=p.selection_reference_size_um * 1e-6,
            min_energy=p.min_impact_energy_j,
            energy_scale=p.energy_to_breakage_factor,
            breakage_gamma=p.breakage_distribution_exponent,
            min_size=p.d_min_um * 1e-6,
            coarse_threshold=p.regime_coarse_threshold_m,
            fine_threshold=p.regime_fine_threshold_m,
            gamma_coarse=p.gamma_coarse,
            gamma_fine=p.gamma_fine,
            clamp_lo_coarse=p.clamp_lo_coarse,
            clamp_hi_coarse=p.clamp_hi_coarse,
            clamp_lo_medium=p.clamp_lo_medium,
            clamp_hi_medium=p.clamp_hi_medium,
            clamp_lo_fine=p.clamp_lo_fine,
            clamp_hi_fine=p.clamp_hi_fine,
        )

        # Copy back
        new_sizes = wp_sizes.numpy()
        new_masses = wp_masses.numpy()
        break_flags = self._wp_break_flags.numpy()

        self._update_stats(sizes, new_sizes, masses, break_flags)

        return new_sizes, new_masses, break_flags

    def step_psd(
        self,
        psd_masses: np.ndarray,
        size_classes: np.ndarray,
        total_impact_energy: float,
        dt: float,
    ) -> np.ndarray:
        """Apply breakage to a particle size distribution.

        Population balance approach for PSD evolution.

        Args:
            psd_masses: Mass in each size class [n_classes]
            size_classes: Size class midpoints [n_classes]
            total_impact_energy: Total impact energy this step
            dt: Timestep [s]

        Returns:
            New PSD masses [n_classes]
        """
        p = self.params

        return breakage_psd_np(
            psd_masses=psd_masses,
            size_classes=size_classes,
            total_impact_energy=total_impact_energy,
            selection_k=p.selection_rate_constant,
            selection_alpha=p.selection_size_exponent,
            breakage_gamma=p.breakage_distribution_exponent,
            dt=dt,
            coarse_threshold=p.regime_coarse_threshold_m,
            fine_threshold=p.regime_fine_threshold_m,
            gamma_coarse=p.gamma_coarse,
            gamma_fine=p.gamma_fine,
        )

    def _update_stats(
        self,
        old_sizes: np.ndarray,
        new_sizes: np.ndarray,
        masses: np.ndarray,
        break_flags: np.ndarray,
    ):
        """Update breakage statistics."""
        num_events = int(break_flags.sum())
        if num_events > 0:
            broken_mask = break_flags == 1
            mass_broken = float(masses[broken_mask].sum())

            old_mean = float(old_sizes[broken_mask].mean())
            new_mean = float(new_sizes[broken_mask].mean())
            reduction = new_mean / old_mean if old_mean > 0 else 1.0

            self._stats = BreakageStats(
                num_breakage_events=num_events,
                total_mass_broken=mass_broken,
                size_reduction_ratio=reduction,
            )
        else:
            self._stats = BreakageStats()

    @property
    def stats(self) -> BreakageStats:
        """Get latest breakage statistics."""
        return self._stats

    def get_size_classes(self) -> np.ndarray:
        """Get size class boundaries in meters."""
        return np.array(self.params.size_classes_m)

    def get_size_class_midpoints(self) -> np.ndarray:
        """Get size class midpoints in meters."""
        boundaries = self.get_size_classes()
        return np.sqrt(boundaries[:-1] * boundaries[1:])  # Geometric mean

    @classmethod
    def from_config(cls, config: "MillConfig", device: str = "cpu") -> "BreakageModel":
        """Create BreakageModel from mill configuration.

        Args:
            config: Mill configuration
            device: "cpu" or "cuda"

        Returns:
            Configured BreakageModel
        """
        return cls(
            params=BreakageParams(),
            device=device,
        )
