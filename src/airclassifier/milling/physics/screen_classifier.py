"""
Screen Classifier
=================

Python wrapper for screen passage classification.
Manages the screen geometry and determines which particles pass through.

The ScreenClassifier:
    - Holds screen geometry parameters
    - Tracks discharged particles
    - Computes product PSD from discharge
    - Manages holdup vs discharge balance
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..kernels import (
    screen_passage_np,
    apply_screen_discharge,
    WARP_AVAILABLE,
)
if WARP_AVAILABLE:
    import warp as wp
    from ..kernels import screen_passage_warp

from ..config import ScreenConfig


@dataclass
class ScreenStats:
    """Statistics from screen classification."""

    num_passed: int = 0
    mass_passed: float = 0.0
    mass_retained: float = 0.0
    passage_rate: float = 0.0  # fraction passed


@dataclass
class ScreenClassifier:
    """Classifier for screen passage in the hammer mill.

    Determines which particles pass through the screen aperture
    and manages the discharge stream.
    """

    # Screen configuration
    config: ScreenConfig = field(default_factory=ScreenConfig)

    # Screen geometry (from housing/screen geometry)
    screen_radius: float = 0.21
    screen_start_angle: float = math.pi  # Start at -Y (bottom)
    screen_arc_angle: float = math.pi    # 180 degrees
    screen_x_start: float = 0.05
    screen_x_end: float = 0.35

    # Device
    device: str = "cpu"

    # Random state
    _rng: Optional[np.random.Generator] = None

    # Discharge buffer
    _discharged_sizes: List[float] = field(default_factory=list)
    _discharged_masses: List[float] = field(default_factory=list)
    _discharged_residence_times: List[float] = field(default_factory=list)

    # Statistics
    _stats: ScreenStats = field(default_factory=ScreenStats)

    # Warp arrays
    _wp_rand_states: Optional["wp.array"] = None
    _wp_passage_flags: Optional["wp.array"] = None

    def __post_init__(self):
        if self._rng is None:
            self._rng = np.random.default_rng()

    def step(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        sizes: np.ndarray,
        masses: np.ndarray,
        residence_times: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Test particles for screen passage.

        Args:
            positions: Particle positions [n, 3]
            velocities: Particle velocities [n, 3]
            sizes: Particle sizes [n]
            masses: Particle masses [n]
            residence_times: Particle residence times [n] (optional)

        Returns:
            (retained_pos, retained_vel, retained_sizes, retained_masses, passage_flags)
        """
        if self.device == "cuda" and WARP_AVAILABLE:
            passage_flags = self._test_passage_warp(positions, velocities, sizes, masses)
        else:
            passage_flags = self._test_passage_np(positions, velocities, sizes, masses)

        # Update stats
        self._update_stats(sizes, masses, passage_flags)

        # Add discharged to buffer
        passed_mask = passage_flags == 1
        self._discharged_sizes.extend(sizes[passed_mask].tolist())
        self._discharged_masses.extend(masses[passed_mask].tolist())
        if residence_times is not None:
            self._discharged_residence_times.extend(residence_times[passed_mask].tolist())

        # Return retained particles
        retained_mask = ~passed_mask
        return (
            positions[retained_mask],
            velocities[retained_mask],
            sizes[retained_mask],
            masses[retained_mask],
            passage_flags,
        )

    def _test_passage_np(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        sizes: np.ndarray,
        masses: np.ndarray,
    ) -> np.ndarray:
        """NumPy passage test."""
        return screen_passage_np(
            positions=positions,
            velocities=velocities,
            sizes=sizes,
            masses=masses,
            screen_radius=self.screen_radius,
            screen_start_angle=self.screen_start_angle,
            screen_arc_angle=self.screen_arc_angle,
            screen_x_start=self.screen_x_start,
            screen_x_end=self.screen_x_end,
            aperture=self.config.aperture_m,
            open_area=self.config.open_area,
            passage_factor=self.config.passage_probability_factor,
            rng=self._rng,
        )

    def _test_passage_warp(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        sizes: np.ndarray,
        masses: np.ndarray,
    ) -> np.ndarray:
        """Warp GPU passage test."""
        n = len(positions)

        # Convert to Warp arrays
        wp_pos = wp.array(positions, dtype=wp.vec3, device="cuda")
        wp_vel = wp.array(velocities, dtype=wp.vec3, device="cuda")
        wp_sizes = wp.array(sizes, dtype=float, device="cuda")
        wp_masses = wp.array(masses, dtype=float, device="cuda")

        # Initialize random states if needed
        if self._wp_rand_states is None or len(self._wp_rand_states) != n:
            rand_seeds = self._rng.integers(0, 2**31, size=n, dtype=np.uint32)
            self._wp_rand_states = wp.array(rand_seeds, dtype=wp.uint32, device="cuda")
            self._wp_passage_flags = wp.zeros(n, dtype=int, device="cuda")

        # Launch kernel
        screen_passage_warp(
            positions=wp_pos,
            velocities=wp_vel,
            sizes=wp_sizes,
            masses=wp_masses,
            passage_flags=self._wp_passage_flags,
            rand_states=self._wp_rand_states,
            screen_radius=self.screen_radius,
            screen_start_angle=self.screen_start_angle,
            screen_arc_angle=self.screen_arc_angle,
            screen_x_start=self.screen_x_start,
            screen_x_end=self.screen_x_end,
            aperture=self.config.aperture_m,
            open_area=self.config.open_area,
            passage_factor=self.config.passage_probability_factor,
        )

        return self._wp_passage_flags.numpy()

    def _update_stats(
        self,
        sizes: np.ndarray,
        masses: np.ndarray,
        passage_flags: np.ndarray,
    ):
        """Update screen statistics."""
        passed_mask = passage_flags == 1
        num_passed = int(passed_mask.sum())
        mass_passed = float(masses[passed_mask].sum())
        mass_retained = float(masses[~passed_mask].sum())
        total_mass = mass_passed + mass_retained

        self._stats = ScreenStats(
            num_passed=num_passed,
            mass_passed=mass_passed,
            mass_retained=mass_retained,
            passage_rate=mass_passed / total_mass if total_mass > 0 else 0.0,
        )

    @property
    def stats(self) -> ScreenStats:
        """Get latest screen statistics."""
        return self._stats

    def get_discharge_psd(
        self,
        size_classes: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Compute PSD of discharged material.

        Args:
            size_classes: Size class boundaries [n+1]

        Returns:
            (mass_fractions [n], total_mass)
        """
        if len(self._discharged_sizes) == 0:
            n = len(size_classes) - 1
            return np.zeros(n), 0.0

        sizes = np.array(self._discharged_sizes)
        masses = np.array(self._discharged_masses)
        total_mass = masses.sum()

        # Bin into size classes
        n = len(size_classes) - 1
        mass_fractions = np.zeros(n)

        for i in range(n):
            mask = (sizes >= size_classes[i]) & (sizes < size_classes[i + 1])
            mass_fractions[i] = masses[mask].sum()

        if total_mass > 0:
            mass_fractions /= total_mass

        return mass_fractions, float(total_mass)

    def get_d_values(self) -> Tuple[float, float, float]:
        """Compute d10, d50, d90 of discharged material.

        Returns:
            (d10, d50, d90) in meters
        """
        if len(self._discharged_sizes) == 0:
            return 0.0, 0.0, 0.0

        sizes = np.array(self._discharged_sizes)
        masses = np.array(self._discharged_masses)

        # Sort by size
        idx = np.argsort(sizes)
        sizes = sizes[idx]
        masses = masses[idx]

        # Cumulative mass fraction
        cumsum = np.cumsum(masses)
        total = cumsum[-1]
        if total == 0:
            return 0.0, 0.0, 0.0

        cumsum /= total

        # Interpolate for d10, d50, d90
        d10 = np.interp(0.10, cumsum, sizes)
        d50 = np.interp(0.50, cumsum, sizes)
        d90 = np.interp(0.90, cumsum, sizes)

        return float(d10), float(d50), float(d90)

    def get_mean_residence_time(self) -> float:
        """Compute mean residence time of discharged particles.

        Returns:
            Mean residence time [s], or 0.0 if no particles discharged.
        """
        if len(self._discharged_residence_times) == 0:
            return 0.0
        return float(np.mean(self._discharged_residence_times))

    def clear_discharge_buffer(self):
        """Clear the discharge buffer."""
        self._discharged_sizes.clear()
        self._discharged_masses.clear()
        self._discharged_residence_times.clear()

    @classmethod
    def from_config(
        cls,
        config: "MillConfig",
        screen_config: Optional[ScreenConfig] = None,
        device: str = "cpu",
    ) -> "ScreenClassifier":
        """Create ScreenClassifier from mill configuration.

        Args:
            config: Mill configuration
            screen_config: Optional screen-specific config
            device: "cpu" or "cuda"

        Returns:
            Configured ScreenClassifier
        """
        if screen_config is None:
            screen_config = ScreenConfig(
                aperture_mm=config.screen_aperture_mm,
                open_area=config.screen_open_area,
            )

        return cls(
            config=screen_config,
            screen_radius=config.screen_inner_radius_m,
            screen_start_angle=math.pi,
            screen_arc_angle=math.radians(config.screen_arc_angle_deg),
            screen_x_start=0.05,
            screen_x_end=0.05 + config.rotor_length_m,
            device=device,
        )
