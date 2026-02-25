"""
Impact Solver
=============

Hammer–particle impact detection and energy transfer for hammer milling kinetics.

The ImpactSolver detects collisions between particles and rotating hammers,
computes impact energies (from relative velocity and restitution), and passes
these to the breakage model. Hammer tip speed (rotor_omega * tip_radius) sets
the impact velocity scale, so breakage is driven by real hammer-mill kinematics.

The ImpactSolver:
    - Manages hammer geometry state (positions, angles)
    - Tracks rotor rotation over time
    - Calls impact detection kernel
    - Accumulates impact statistics (energy for breakage)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..kernels import (
    impact_detection_np,
    WARP_AVAILABLE,
)
if WARP_AVAILABLE:
    import warp as wp
    from ..kernels import impact_detection_warp


@dataclass
class ImpactStats:
    """Statistics from impact detection."""

    num_impacts: int = 0
    total_impact_energy: float = 0.0
    max_impact_energy: float = 0.0
    mean_impact_energy: float = 0.0


@dataclass
class ImpactSolver:
    """Solver for hammer-particle impacts.

    Manages the hammer geometry and performs collision detection
    against particles in the mill chamber.
    """

    # Hammer configuration
    hammer_tip_radius: float = 0.16
    hammer_width: float = 0.05
    hammer_rows: int = 4
    hammers_per_row: int = 4
    row_start_x: float = 0.08
    row_spacing: float = 0.06

    # Physics parameters
    restitution: float = 0.3

    # State
    _rotor_theta: float = 0.0
    _rotor_omega: float = 0.0
    _stats: ImpactStats = field(default_factory=ImpactStats)

    # Device
    device: str = "cpu"

    # Warp arrays (lazy init)
    _wp_impact_flags: Optional["wp.array"] = None
    _wp_impact_energies: Optional["wp.array"] = None

    def set_rotor_state(self, theta: float, omega: float):
        """Set the current rotor state.

        Args:
            theta: Current rotor angle [rad]
            omega: Rotor angular velocity [rad/s]
        """
        self._rotor_theta = theta
        self._rotor_omega = omega

    def step(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        sizes: np.ndarray,
        masses: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect and resolve impacts.

        Args:
            positions: Particle positions [n, 3]
            velocities: Particle velocities [n, 3] (modified in place)
            sizes: Particle sizes [n]
            masses: Particle masses [n]
            dt: Timestep [s]

        Returns:
            (impact_flags, impact_energies, new_velocities)
        """
        n = len(positions)

        if self.device == "cuda" and WARP_AVAILABLE:
            return self._step_warp(positions, velocities, sizes, masses, dt)
        else:
            return self._step_np(positions, velocities, sizes, masses, dt)

    def _step_np(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        sizes: np.ndarray,
        masses: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """NumPy implementation."""
        impact_flags, impact_energies, new_velocities = impact_detection_np(
            positions=positions,
            velocities=velocities,
            sizes=sizes,
            masses=masses,
            rotor_theta=self._rotor_theta,
            rotor_omega=self._rotor_omega,
            hammer_tip_radius=self.hammer_tip_radius,
            hammer_width=self.hammer_width,
            hammer_rows=self.hammer_rows,
            hammers_per_row=self.hammers_per_row,
            row_start_x=self.row_start_x,
            row_spacing=self.row_spacing,
            restitution=self.restitution,
            dt=dt,
        )

        # Update stats
        self._update_stats(impact_flags, impact_energies)

        return impact_flags, impact_energies, new_velocities

    def _step_warp(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        sizes: np.ndarray,
        masses: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Warp GPU implementation."""
        n = len(positions)

        # Convert to Warp arrays
        wp_pos = wp.array(positions, dtype=wp.vec3, device="cuda")
        wp_vel = wp.array(velocities, dtype=wp.vec3, device="cuda")
        wp_sizes = wp.array(sizes, dtype=float, device="cuda")
        wp_masses = wp.array(masses, dtype=float, device="cuda")

        # Allocate output arrays
        if self._wp_impact_flags is None or len(self._wp_impact_flags) != n:
            self._wp_impact_flags = wp.zeros(n, dtype=int, device="cuda")
            self._wp_impact_energies = wp.zeros(n, dtype=float, device="cuda")

        # Launch kernel
        impact_detection_warp(
            positions=wp_pos,
            velocities=wp_vel,
            sizes=wp_sizes,
            masses=wp_masses,
            impact_flags=self._wp_impact_flags,
            impact_energies=self._wp_impact_energies,
            rotor_theta=self._rotor_theta,
            rotor_omega=self._rotor_omega,
            hammer_tip_radius=self.hammer_tip_radius,
            hammer_width=self.hammer_width,
            hammer_rows=self.hammer_rows,
            hammers_per_row=self.hammers_per_row,
            row_start_x=self.row_start_x,
            row_spacing=self.row_spacing,
            restitution=self.restitution,
            dt=dt,
        )

        # Copy back to CPU
        impact_flags = self._wp_impact_flags.numpy()
        impact_energies = self._wp_impact_energies.numpy()
        new_velocities = wp_vel.numpy()

        # Update stats
        self._update_stats(impact_flags, impact_energies)

        return impact_flags, impact_energies, new_velocities

    def _update_stats(self, impact_flags: np.ndarray, impact_energies: np.ndarray):
        """Update impact statistics."""
        num_impacts = int(impact_flags.sum())
        if num_impacts > 0:
            energies = impact_energies[impact_flags == 1]
            self._stats = ImpactStats(
                num_impacts=num_impacts,
                total_impact_energy=float(energies.sum()),
                max_impact_energy=float(energies.max()),
                mean_impact_energy=float(energies.mean()),
            )
        else:
            self._stats = ImpactStats()

    @property
    def stats(self) -> ImpactStats:
        """Get latest impact statistics."""
        return self._stats

    def compute_power_draw(self, dt: float) -> float:
        """Compute power draw from impacts.

        Args:
            dt: Timestep [s]

        Returns:
            Power [W] dissipated in impacts
        """
        if dt > 0:
            return self._stats.total_impact_energy / dt
        return 0.0

    @classmethod
    def from_config(cls, config: "MillConfig", device: str = "cpu") -> "ImpactSolver":
        """Create ImpactSolver from mill configuration.

        Args:
            config: Mill configuration
            device: "cpu" or "cuda"

        Returns:
            Configured ImpactSolver
        """
        from ..config import MillConfig

        # Compute hammer tip radius
        tip_radius = config.rotor_diameter_m / 2.0 + config.hammer_length_m

        # Compute row spacing
        margin = 0.04
        usable_length = config.rotor_length_m - 2 * margin
        spacing = usable_length / max(config.hammer_rows - 1, 1)

        return cls(
            hammer_tip_radius=tip_radius,
            hammer_width=config.hammer_width_m,
            hammer_rows=config.hammer_rows,
            hammers_per_row=config.hammers_per_row,
            row_start_x=margin + 0.05,
            row_spacing=spacing,
            restitution=0.3,
            device=device,
        )
