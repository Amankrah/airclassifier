"""
Convergence Detector
====================

Detects steady-state conditions and target achievements for
physics-based simulation termination.

Monitors rolling windows of key performance indicators (KPIs)
to determine when the simulation has reached equilibrium or
achieved target conditions.

Termination modes:
    - time: Run for fixed duration (default, legacy behavior)
    - mass: Run until target mass has been discharged
    - steady_state: Run until KPIs stabilize (d50, throughput)
    - target_d50: Run until target median particle size achieved
    - batch_complete: Run until all fed material is discharged (for batch processing)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

import numpy as np


@dataclass
class ConvergenceDetector:
    """Detects steady-state and target conditions for simulation termination.

    Tracks rolling windows of KPIs and evaluates termination criteria
    based on the configured mode.

    Attributes:
        mode: Termination mode ("time", "mass", "steady_state", "target_d50")
        window_size: Number of steps for rolling statistics
        d50_stability_threshold: Max coefficient of variation for d50 stability
        throughput_stability_threshold: Max CV for throughput stability
        target_d50_um: Target median particle size [µm]
        target_mass_kg: Target discharged mass [kg]
        min_run_time_s: Minimum simulation time before early termination
        max_run_time_s: Maximum simulation time (safety limit)
    """

    # Termination mode
    mode: str = "time"

    # Rolling window for variance calculation
    window_size: int = 50  # ~100ms at dt=2ms

    # Stability thresholds (coefficient of variation)
    d50_stability_threshold: float = 0.05      # 5% variation
    throughput_stability_threshold: float = 0.10  # 10% variation

    # Target values
    target_d50_um: float = 0.0
    target_d50_tolerance_pct: float = 5.0  # Within 5% of target
    target_mass_kg: float = 0.0

    # Time limits
    min_run_time_s: float = 5.0
    max_run_time_s: float = 300.0  # Safety limit

    # Internal state (rolling windows)
    _d50_history: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    _throughput_history: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    _power_history: Deque[float] = field(default_factory=lambda: deque(maxlen=50))

    # Cumulative tracking
    _cumulative_discharge_kg: float = 0.0
    _cumulative_feed_kg: float = 0.0  # Track total mass fed for batch mode
    _current_time_s: float = 0.0
    _last_dt: float = 0.002

    # Batch mode tracking
    _target_feed_mass_kg: float = 0.0  # Target batch size (0 = continuous)
    _feeding_complete: bool = False  # True when all batch material has been fed

    # Particle tracking for empty mill detection
    _current_particle_count: int = 0
    _empty_mill_steps: int = 0  # Consecutive steps with zero particles
    _had_particles: bool = False  # True once particles have been fed

    def __post_init__(self):
        """Initialize deques with correct maxlen."""
        self._d50_history = deque(maxlen=self.window_size)
        self._throughput_history = deque(maxlen=self.window_size)
        self._power_history = deque(maxlen=self.window_size)

    def reset(self):
        """Reset detector state for new simulation."""
        self._d50_history.clear()
        self._throughput_history.clear()
        self._power_history.clear()
        self._cumulative_discharge_kg = 0.0
        self._cumulative_feed_kg = 0.0
        self._current_time_s = 0.0
        self._current_particle_count = 0
        self._empty_mill_steps = 0
        self._had_particles = False
        self._feeding_complete = False

    def update(
        self,
        time_s: float,
        d50_m: float,
        discharge_rate_kg_per_s: float,
        power_kw: float,
        dt: float,
        particle_count: int = 0,
        feed_rate_kg_per_s: float = 0.0,
    ) -> None:
        """Update detector with latest simulation state.

        Args:
            time_s: Current simulation time [s]
            d50_m: Current median particle size [m]
            discharge_rate_kg_per_s: Current discharge rate [kg/s]
            power_kw: Current power draw [kW]
            dt: Timestep [s]
            particle_count: Number of particles currently in mill chamber
            feed_rate_kg_per_s: Current feed rate [kg/s] (for batch tracking)
        """
        self._current_time_s = time_s
        self._last_dt = dt

        # Track particle count for empty mill detection
        self._current_particle_count = particle_count
        if particle_count > 0:
            self._had_particles = True
            self._empty_mill_steps = 0
        elif self._had_particles:
            # Mill was fed but now empty - count consecutive empty steps
            self._empty_mill_steps += 1

        # Track cumulative feed for batch mode
        self._cumulative_feed_kg += feed_rate_kg_per_s * dt

        # Detect when batch feeding is complete
        if self._target_feed_mass_kg > 0 and not self._feeding_complete:
            if self._cumulative_feed_kg >= self._target_feed_mass_kg * 0.99:
                self._feeding_complete = True

        # Convert d50 to micrometers for comparison
        d50_um = d50_m * 1e6

        # Update rolling windows
        if d50_um > 0:  # Only track valid d50 values
            self._d50_history.append(d50_um)
        self._throughput_history.append(discharge_rate_kg_per_s * 3600)  # kg/hr
        self._power_history.append(power_kw)

        # Track cumulative discharge
        self._cumulative_discharge_kg += discharge_rate_kg_per_s * dt

    def is_steady_state(self) -> bool:
        """Check if simulation has reached steady-state.

        Returns:
            True if all tracked KPIs are stable within thresholds.
        """
        # Need full window of data
        if len(self._d50_history) < self.window_size:
            return False

        # Coefficient of variation for d50
        d50_arr = np.array(self._d50_history)
        d50_mean = np.mean(d50_arr)
        if d50_mean > 0:
            d50_cv = np.std(d50_arr) / d50_mean
        else:
            return False  # No valid d50 data

        # Coefficient of variation for throughput
        throughput_arr = np.array(self._throughput_history)
        throughput_mean = np.mean(throughput_arr)
        if throughput_mean > 0:
            throughput_cv = np.std(throughput_arr) / throughput_mean
        else:
            throughput_cv = 0.0  # Zero throughput is stable

        return (
            d50_cv < self.d50_stability_threshold and
            throughput_cv < self.throughput_stability_threshold
        )

    def is_target_d50_reached(self) -> bool:
        """Check if target d50 has been achieved.

        Returns:
            True if current d50 is within tolerance of target.
        """
        if self.target_d50_um <= 0:
            return False

        if len(self._d50_history) < 10:  # Need some data
            return False

        # Use recent average to smooth noise
        recent_d50 = np.mean(list(self._d50_history)[-10:])
        tolerance = self.target_d50_um * self.target_d50_tolerance_pct / 100.0

        return abs(recent_d50 - self.target_d50_um) <= tolerance

    def is_mass_target_reached(self) -> bool:
        """Check if target discharge mass has been processed.

        Returns:
            True if cumulative discharge >= target mass.
        """
        if self.target_mass_kg <= 0:
            return False

        return self._cumulative_discharge_kg >= self.target_mass_kg

    def is_mill_empty(self) -> bool:
        """Check if mill chamber is empty (all material discharged).

        Returns:
            True if mill had particles but is now empty for several steps.
        """
        # Mill must have received particles at some point
        if not self._had_particles:
            return False

        # Need several consecutive empty steps to confirm (avoid false positives)
        # At dt=0.002s, 25 steps = 50ms of confirmed empty state
        return self._empty_mill_steps >= 25

    def should_terminate(self) -> Tuple[bool, str]:
        """Check if simulation should terminate based on current mode.

        Returns:
            (should_stop, reason) tuple where reason explains termination.
        """
        # Always check max time safety limit
        if self._current_time_s >= self.max_run_time_s:
            return True, f"Max time reached ({self.max_run_time_s:.0f}s)"

        # Time-based mode: never early terminate (handled externally)
        if self.mode == "time":
            return False, ""

        # All physics-based modes require minimum run time
        if self._current_time_s < self.min_run_time_s:
            return False, ""

        # Check for empty mill (all material discharged) - applies to all physics modes
        if self.is_mill_empty():
            discharged = self._cumulative_discharge_kg
            return True, f"Mill empty - all material discharged ({discharged:.3f} kg)"

        # Mass-processed mode
        if self.mode == "mass":
            if self.is_mass_target_reached():
                return True, f"Target mass reached ({self.target_mass_kg:.2f} kg)"
            return False, ""

        # Steady-state mode
        if self.mode == "steady_state":
            if self.is_steady_state():
                d50_mean = np.mean(self._d50_history) if self._d50_history else 0
                return True, f"Steady-state reached (d50={d50_mean:.0f} µm)"
            return False, ""

        # Target d50 mode
        if self.mode == "target_d50":
            if self.is_target_d50_reached():
                recent_d50 = np.mean(list(self._d50_history)[-10:]) if self._d50_history else 0
                return True, f"Target d50 reached ({recent_d50:.0f} µm)"
            return False, ""

        # Batch complete mode: run until all fed material is discharged
        if self.mode == "batch_complete":
            if self._feeding_complete and self.is_mill_empty():
                return True, f"Batch complete ({self._cumulative_discharge_kg:.3f} kg discharged)"
            return False, ""

        return False, ""

    @property
    def current_d50_um(self) -> float:
        """Get current (smoothed) d50 in micrometers."""
        if len(self._d50_history) >= 5:
            return float(np.mean(list(self._d50_history)[-5:]))
        elif self._d50_history:
            return float(self._d50_history[-1])
        return 0.0

    @property
    def current_throughput_kg_hr(self) -> float:
        """Get current (smoothed) throughput in kg/hr."""
        if len(self._throughput_history) >= 5:
            return float(np.mean(list(self._throughput_history)[-5:]))
        elif self._throughput_history:
            return float(self._throughput_history[-1])
        return 0.0

    @property
    def cumulative_discharge_kg(self) -> float:
        """Get total discharged mass."""
        return self._cumulative_discharge_kg

    @property
    def cumulative_feed_kg(self) -> float:
        """Get total fed mass."""
        return self._cumulative_feed_kg

    @property
    def feeding_complete(self) -> bool:
        """Check if batch feeding is complete."""
        return self._feeding_complete

    def set_target_feed_mass(self, mass_kg: float) -> None:
        """Set target batch feed mass for batch_complete mode.

        Args:
            mass_kg: Target mass to feed [kg]. Set to 0 for continuous mode.
        """
        self._target_feed_mass_kg = mass_kg
        self._feeding_complete = False

    @property
    def progress_pct(self) -> float:
        """Get progress percentage based on mode.

        Returns:
            Progress as percentage (0-100).

        For batch_complete mode:
            - 0-50%: Feeding phase (material entering the mill)
            - 50-100%: Discharge phase (material leaving the mill)
        """
        if self.mode == "batch_complete" and self._target_feed_mass_kg > 0:
            # Batch mode: feeding (0-50%) + discharge (50-100%)
            feed_progress = min(1.0, self._cumulative_feed_kg / self._target_feed_mass_kg)

            if not self._feeding_complete:
                # Still in feeding phase: 0-50%
                return feed_progress * 50.0
            else:
                # In discharge phase: 50-100%
                # Progress based on how much has been discharged vs fed
                if self._cumulative_feed_kg > 0:
                    discharge_progress = min(1.0, self._cumulative_discharge_kg / self._cumulative_feed_kg)
                    return 50.0 + discharge_progress * 50.0
                return 50.0

        elif self.mode == "mass" and self.target_mass_kg > 0:
            return min(100.0, 100.0 * self._cumulative_discharge_kg / self.target_mass_kg)
        elif self.mode == "target_d50" and self.target_d50_um > 0:
            if self._d50_history:
                current = self.current_d50_um
                # Progress from feed size (assume 3000 µm) toward target
                feed_d50 = 3000.0
                total_reduction = feed_d50 - self.target_d50_um
                achieved = feed_d50 - current
                if total_reduction > 0:
                    return min(100.0, max(0.0, 100.0 * achieved / total_reduction))
            return 0.0
        elif self.mode == "steady_state":
            # Estimate based on window fill and variance reduction
            fill_pct = min(100.0, 100.0 * len(self._d50_history) / self.window_size)
            if len(self._d50_history) >= self.window_size:
                d50_arr = np.array(self._d50_history)
                d50_mean = np.mean(d50_arr)
                if d50_mean > 0:
                    d50_cv = np.std(d50_arr) / d50_mean
                    stability_pct = max(0.0, 100.0 * (1.0 - d50_cv / self.d50_stability_threshold))
                    return min(100.0, 0.5 * fill_pct + 0.5 * stability_pct)
            return 0.5 * fill_pct
        else:
            # Time mode: no inherent progress
            return 0.0


@dataclass
class TerminationConfig:
    """Configuration for simulation termination criteria.

    This is used to configure the ConvergenceDetector and is
    stored as part of the recipe.
    """

    mode: str = "time"  # "time", "mass", "steady_state", "target_d50", "batch_complete"
    run_duration_s: float = 60.0
    target_mass_kg: float = 1.0
    target_d50_um: float = 500.0
    target_feed_mass_kg: float = 0.0  # For batch_complete mode (0 = continuous)
    min_run_time_s: float = 5.0
    max_run_time_s: float = 300.0
    steady_state_window: int = 50
    d50_tolerance_pct: float = 5.0
    throughput_tolerance_pct: float = 10.0

    def create_detector(self) -> ConvergenceDetector:
        """Create a ConvergenceDetector from this config."""
        detector = ConvergenceDetector(
            mode=self.mode,
            window_size=self.steady_state_window,
            d50_stability_threshold=self.d50_tolerance_pct / 100.0,
            throughput_stability_threshold=self.throughput_tolerance_pct / 100.0,
            target_d50_um=self.target_d50_um,
            target_d50_tolerance_pct=self.d50_tolerance_pct,
            target_mass_kg=self.target_mass_kg,
            min_run_time_s=self.min_run_time_s,
            max_run_time_s=self.max_run_time_s,
        )
        # Set batch target if configured
        if self.target_feed_mass_kg > 0:
            detector.set_target_feed_mass(self.target_feed_mass_kg)
        return detector
