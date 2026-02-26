"""
Hammer Mill Simulator
=====================

High-level simulator class that wraps geometry and physics
for the hammer mill. Provides a clean API for running simulations
and extracting results.

This is the main entry point for using the milling module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import (
    MillConfig,
    MillRecipe,
    ScreenConfig,
    BreakageParams,
    MillingOutletState,
)
from .geometry import (
    create_hammer_mill_machine,
    HammerMillMachineAssembly,
    COMPONENT_COLORS,
)
from .physics import (
    CoupledMillingEngine,
    MillingStepState,
    ParticleState,
    TerminationConfig,
)


@dataclass
class MillingResult:
    """Results from a milling simulation run."""

    # Time series
    history: List[MillingStepState] = field(default_factory=list)

    # Final PSD (discharged / passed through screen)
    psd_size_classes_m: np.ndarray = field(default_factory=lambda: np.array([]))
    psd_mass_fractions: np.ndarray = field(default_factory=lambda: np.array([]))
    psd_total_mass_kg: float = 0.0

    # Retained PSD (particles still in chamber / too large to pass)
    retained_psd_mass_fractions: np.ndarray = field(default_factory=lambda: np.array([]))
    retained_psd_size_classes_m: np.ndarray = field(default_factory=lambda: np.array([]))
    retained_total_mass_kg: float = 0.0

    # Particle tracking: passed through screen (fine powder) vs retained in mill
    total_particles_passed: int = 0
    total_particles_retained: int = 0
    holdup_kg_final: float = 0.0
    total_particles_fed: int = 0
    total_mass_fed_kg: float = 0.0

    # Breakage tracking
    total_breakage_events: int = 0
    total_impacts: int = 0

    # Summary statistics — flour (discharged through screen)
    d10_um: float = 0.0
    d50_um: float = 0.0
    d90_um: float = 0.0

    # Retained seed d-values
    retained_d10_um: float = 0.0
    retained_d50_um: float = 0.0
    retained_d90_um: float = 0.0
    throughput_kg_per_hr: float = 0.0
    mean_residence_time_s: float = 0.0
    specific_energy_kwh_per_t: float = 0.0

    # Power
    mean_power_kw: float = 0.0
    max_power_kw: float = 0.0

    def to_outlet_state(
        self,
        avg_temperature_c: Optional[float] = None,
        avg_moisture_wb: Optional[float] = None,
    ) -> MillingOutletState:
        """Convert to MillingOutletState for pipeline integration.

        Args:
            avg_temperature_c: Passthrough from feed (e.g. pretreatment outlet). Default 25.0.
            avg_moisture_wb: Passthrough from feed. Default 0.12.
        """
        T = avg_temperature_c if avg_temperature_c is not None else 25.0
        M = avg_moisture_wb if avg_moisture_wb is not None else 0.12
        return MillingOutletState(
            psd_mass_fractions=tuple(self.psd_mass_fractions.tolist()),
            psd_size_classes_um=tuple((self.psd_size_classes_m * 1e6).tolist()),
            d10_um=self.d10_um,
            d50_um=self.d50_um,
            d90_um=self.d90_um,
            throughput_kg_per_hr=self.throughput_kg_per_hr,
            mass_holdup_kg=self.history[-1].holdup_kg if self.history else 0.0,
            avg_temperature_c=T,
            avg_moisture_wb=M,
            power_kw=self.mean_power_kw,
            specific_energy_kwh_per_t=self.specific_energy_kwh_per_t,
            mean_residence_time_s=self.mean_residence_time_s,
        )


@dataclass
class HammerMillSimulator:
    """High-level hammer mill simulator.

    Combines geometry and physics into a single interface.
    Provides methods for configuration, running simulations,
    and extracting results.
    """

    # Configuration
    config: MillConfig = field(default_factory=MillConfig)

    # Geometry assembly
    assembly: Optional[HammerMillMachineAssembly] = None

    # Physics engine
    engine: Optional[CoupledMillingEngine] = None

    # Current recipe
    recipe: Optional[MillRecipe] = None

    # Device
    device: str = "cpu"

    # Results
    _last_result: Optional[MillingResult] = None

    def __post_init__(self):
        """Initialize geometry and physics."""
        self._build_assembly()
        self._build_engine()

    def _build_assembly(self):
        """Build the geometry assembly."""
        self.assembly = create_hammer_mill_machine(
            config=self.config,
            build_meshes=True,
        )

    def _build_engine(self):
        """Build the physics engine."""
        self.engine = CoupledMillingEngine(
            config=self.config,
            device=self.device,
        )

    def load_recipe(self, recipe: MillRecipe):
        """Load a milling recipe.

        Args:
            recipe: Recipe with operating parameters
        """
        self.recipe = recipe
        self.engine.load_recipe(recipe)

    def initialize(
        self,
        initial_holdup_kg: float = 0.0,
        initial_particles: int = 0,
    ):
        """Initialize the simulation.

        Args:
            initial_holdup_kg: Initial mass in mill
            initial_particles: Number of initial particles
        """
        if self.recipe is None:
            self.recipe = MillRecipe()
            self.engine.load_recipe(self.recipe)

        self.engine.initialize(
            initial_holdup_kg=initial_holdup_kg,
            initial_particles=initial_particles,
        )

    def step(self, dt: float = 0.001) -> MillingStepState:
        """Advance simulation by one timestep.

        Args:
            dt: Timestep [s]

        Returns:
            State at end of step
        """
        return self.engine.step(dt)

    def set_termination_config(self, config: "TerminationConfig") -> None:
        """Set termination configuration for physics-based stopping.

        Args:
            config: Termination configuration
        """
        self.engine.set_termination_config(config)

    def check_termination(self) -> tuple:
        """Check if simulation should terminate based on physics criteria.

        Returns:
            (should_stop, reason) tuple
        """
        return self.engine.check_termination()

    def get_convergence_progress(self) -> float:
        """Get progress percentage for physics-based termination modes.

        Returns:
            Progress as percentage (0-100).
        """
        return self.engine.get_convergence_progress()

    def run(
        self,
        duration_s: Optional[float] = None,
        dt: float = 0.001,
    ) -> MillingResult:
        """Run simulation for a duration.

        Args:
            duration_s: Duration [s]. If None, uses recipe.run_duration_s
            dt: Timestep [s]

        Returns:
            MillingResult with time series and summary statistics
        """
        if duration_s is None:
            if self.recipe is not None:
                duration_s = self.recipe.run_duration_s
            else:
                duration_s = 60.0

        # Run simulation
        states = self.engine.run(duration_s, dt)

        # Extract PSD (discharged)
        size_classes, mass_fractions, total_mass = self.engine.get_discharge_psd()
        # Prefer mass-balance total (see build_result_from_engine comment)
        if len(states) > 0:
            mass_balance_total = sum(s.discharge_rate_kg_per_s * dt for s in states)
            total_mass = max(total_mass, mass_balance_total)

        # Extract retained PSD (still in chamber)
        _, retained_fractions, retained_mass = self.engine.get_retained_psd()

        # Compute summary statistics
        d10, d50, d90 = self.engine.screen_classifier.get_d_values()

        # Compute retained seed d-values
        ret_sizes = self.engine.particles.sizes
        ret_masses = self.engine.particles.masses
        if len(ret_sizes) > 0 and ret_masses.sum() > 0:
            idx = np.argsort(ret_sizes)
            s_sorted = ret_sizes[idx]
            m_sorted = ret_masses[idx]
            cum = np.cumsum(m_sorted)
            cum /= cum[-1]
            ret_d10 = float(np.interp(0.10, cum, s_sorted)) * 1e6
            ret_d50 = float(np.interp(0.50, cum, s_sorted)) * 1e6
            ret_d90 = float(np.interp(0.90, cum, s_sorted)) * 1e6
        else:
            ret_d10 = ret_d50 = ret_d90 = 0.0

        # Compute averages from history
        if len(states) > 0:
            powers = [s.power_kw for s in states]
            mean_power = np.mean(powers)
            max_power = np.max(powers)

            # Throughput (discharge rate)
            discharge_rates = [s.discharge_rate_kg_per_s * 3600 for s in states]  # kg/hr
            throughput = np.mean(discharge_rates)

            # Residence time (mean of particles that passed)
            # Approximate from feed/discharge balance
            if len(states) > 1:
                total_fed = sum(s.num_fed for s in states)
                total_discharged = sum(s.num_discharged for s in states)
                if total_fed > 0:
                    mean_residence = duration_s * (1 - total_discharged / max(total_fed, 1))
                else:
                    mean_residence = 0.0
            else:
                mean_residence = 0.0

            # Specific energy
            total_energy_kwh = sum(s.power_kw * dt / 3600 for s in states)
            total_throughput_t = total_mass / 1000  # tonnes
            if total_throughput_t > 0:
                specific_energy = total_energy_kwh / total_throughput_t
            else:
                specific_energy = 0.0
        else:
            mean_power = 0.0
            max_power = 0.0
            throughput = 0.0
            mean_residence = 0.0
            specific_energy = 0.0

        total_passed = sum(s.num_passed_screen for s in states)
        total_fed = sum(s.num_fed for s in states)
        total_mass_fed_kg = sum(s.feed_rate_kg_per_s * dt for s in states) if len(states) > 0 else 0.0
        total_breakage = sum(s.num_breakage_events for s in states)
        total_impacts_sum = sum(s.num_impacts for s in states)
        last_state = states[-1] if states else None
        holdup_final = last_state.holdup_kg if last_state else 0.0
        retained_count = last_state.num_particles if last_state else 0

        result = MillingResult(
            history=states,
            psd_size_classes_m=size_classes,
            psd_mass_fractions=mass_fractions,
            psd_total_mass_kg=total_mass,
            retained_psd_mass_fractions=retained_fractions,
            retained_psd_size_classes_m=size_classes.copy(),
            retained_total_mass_kg=retained_mass,
            total_particles_passed=total_passed,
            total_particles_retained=retained_count,
            holdup_kg_final=holdup_final,
            total_particles_fed=total_fed,
            total_mass_fed_kg=total_mass_fed_kg,
            total_breakage_events=total_breakage,
            total_impacts=total_impacts_sum,
            d10_um=d10 * 1e6,
            d50_um=d50 * 1e6,
            d90_um=d90 * 1e6,
            retained_d10_um=ret_d10,
            retained_d50_um=ret_d50,
            retained_d90_um=ret_d90,
            throughput_kg_per_hr=throughput,
            mean_residence_time_s=mean_residence,
            specific_energy_kwh_per_t=specific_energy,
            mean_power_kw=mean_power,
            max_power_kw=max_power,
        )

        self._last_result = result
        return result

    def build_result_from_engine(self, duration_s: float, dt: float = 0.001) -> MillingResult:
        """Build MillingResult from current engine state (e.g. after stepping in UI).

        Args:
            duration_s: Simulated duration (for reporting).
            dt: Timestep used.

        Returns:
            MillingResult populated from engine history and discharge PSD.
        """
        states = self.engine.history
        size_classes, mass_fractions, total_mass = self.engine.get_discharge_psd()
        # The Lagrangian breakage model does not conserve individual particle mass
        # (each break shrinks the tracked particle, "losing" mass to untracked fines).
        # The mass-balance discharge rate (pre_holdup + feed - post_holdup) is physically
        # correct, so always prefer it over the particle-level buffer sum.
        if len(states) > 0:
            mass_balance_total = sum(s.discharge_rate_kg_per_s * dt for s in states)
            total_mass = max(total_mass, mass_balance_total)
        retained_size_classes, retained_fractions, retained_mass = self.engine.get_retained_psd()
        d10, d50, d90 = self.engine.screen_classifier.get_d_values()

        # Compute retained seed d-values from current particle state
        ret_sizes = self.engine.particles.sizes
        ret_masses = self.engine.particles.masses
        if len(ret_sizes) > 0 and ret_masses.sum() > 0:
            idx = np.argsort(ret_sizes)
            s_sorted = ret_sizes[idx]
            m_sorted = ret_masses[idx]
            cum = np.cumsum(m_sorted)
            cum /= cum[-1]
            ret_d10 = float(np.interp(0.10, cum, s_sorted)) * 1e6
            ret_d50 = float(np.interp(0.50, cum, s_sorted)) * 1e6
            ret_d90 = float(np.interp(0.90, cum, s_sorted)) * 1e6
        else:
            ret_d10 = ret_d50 = ret_d90 = 0.0

        if len(states) > 0:
            powers = [s.power_kw for s in states]
            mean_power = float(np.mean(powers))
            max_power = float(np.max(powers))
            discharge_rates = [s.discharge_rate_kg_per_s * 3600 for s in states]
            throughput = float(np.mean(discharge_rates))
            total_fed = sum(s.num_fed for s in states)
            total_mass_fed_kg = sum(s.feed_rate_kg_per_s * dt for s in states)
            total_discharged = sum(s.num_discharged for s in states)
            mean_residence = (
                duration_s * (1 - total_discharged / max(total_fed, 1))
                if total_fed > 0 and len(states) > 1
                else 0.0
            )
            total_energy_kwh = sum(s.power_kw * dt / 3600 for s in states)
            total_throughput_t = total_mass / 1000
            specific_energy = total_energy_kwh / total_throughput_t if total_throughput_t > 0 else 0.0
            total_passed = sum(s.num_passed_screen for s in states)
            total_breakage = sum(s.num_breakage_events for s in states)
            total_impacts_sum = sum(s.num_impacts for s in states)
            last_state = states[-1]
            holdup_final = last_state.holdup_kg
            retained_count = last_state.num_particles
        else:
            mean_power = max_power = throughput = mean_residence = specific_energy = 0.0
            total_passed = total_fed = total_breakage = total_impacts_sum = 0
            total_mass_fed_kg = 0.0
            holdup_final = 0.0
            retained_count = 0

        result = MillingResult(
            history=states,
            psd_size_classes_m=size_classes,
            psd_mass_fractions=mass_fractions,
            psd_total_mass_kg=total_mass,
            retained_psd_mass_fractions=retained_fractions,
            retained_psd_size_classes_m=retained_size_classes,
            retained_total_mass_kg=retained_mass,
            total_particles_passed=total_passed,
            total_particles_retained=retained_count,
            holdup_kg_final=holdup_final,
            total_particles_fed=total_fed,
            total_mass_fed_kg=total_mass_fed_kg,
            total_breakage_events=total_breakage,
            total_impacts=total_impacts_sum,
            d10_um=d10 * 1e6,
            d50_um=d50 * 1e6,
            d90_um=d90 * 1e6,
            retained_d10_um=ret_d10,
            retained_d50_um=ret_d50,
            retained_d90_um=ret_d90,
            throughput_kg_per_hr=throughput,
            mean_residence_time_s=mean_residence,
            specific_energy_kwh_per_t=specific_energy,
            mean_power_kw=mean_power,
            max_power_kw=max_power,
        )
        self._last_result = result
        return result

    def get_outlet_conditions(self) -> MillingOutletState:
        """Get outlet conditions for pipeline integration.

        Returns:
            MillingOutletState with PSD and flow data
        """
        if self._last_result is not None and self.engine is not None:
            return self._last_result.to_outlet_state(
                avg_temperature_c=self.engine._inlet_temperature_c,
                avg_moisture_wb=self.engine._inlet_moisture_wb,
            )

        # Return empty state if no simulation run yet
        return MillingOutletState()

    def set_inlet_state(self, outlet_state: Any) -> None:
        """Set feed from pretreatment outlet (pipeline integration).

        Args:
            outlet_state: OutletState from airclassifier.pretreatment (GP-15 outfeed).
                Must have throughput_kg_per_hr, avg_temperature_c, avg_moisture_wb.
        """
        if self.engine is None:
            return
        self.engine.set_inlet_state(outlet_state)

    def get_geometry_meshes(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, dict]]:
        """Get geometry meshes for visualization.

        Returns:
            Dictionary mapping component name to (vertices, triangles, metadata)
        """
        if self.assembly is None:
            self._build_assembly()
        return self.assembly.get_component_meshes()

    def get_animated_components(self) -> Dict[str, dict]:
        """Get components that need animation.

        Returns:
            Dictionary mapping component name to animation metadata
        """
        if self.assembly is None:
            self._build_assembly()
        return self.assembly.get_animated_components()

    def get_rotor_angle(self) -> float:
        """Get current rotor angle [rad].

        Returns:
            Rotor angle for animation
        """
        return self.engine.rotor_theta

    def get_particle_positions(self) -> np.ndarray:
        """Get current particle positions.

        Returns:
            Array of [n, 3] positions
        """
        return self.engine.particles.positions

    def get_particle_sizes(self) -> np.ndarray:
        """Get current particle sizes.

        Returns:
            Array of [n] sizes
        """
        return self.engine.particles.sizes

    def get_all_visible_particles(self) -> tuple:
        """Get all visible particles including discharge flow.

        Returns:
            (positions [n, 3], sizes [n]) including chamber and discharge particles
        """
        return self.engine.get_all_visible_particles()

    @property
    def history(self) -> List[MillingStepState]:
        """Get simulation history."""
        return self.engine.history

    @classmethod
    def create(
        cls,
        config: Optional[MillConfig] = None,
        recipe: Optional[MillRecipe] = None,
        device: str = "cpu",
    ) -> "HammerMillSimulator":
        """Factory method to create a configured simulator.

        Args:
            config: Mill configuration (uses defaults if None)
            recipe: Operating recipe (uses defaults if None)
            device: "cpu" or "cuda"

        Returns:
            Configured HammerMillSimulator
        """
        config = config or MillConfig()
        sim = cls(config=config, device=device)

        if recipe is not None:
            sim.load_recipe(recipe)

        return sim


def run_milling_simulation(
    config: Optional[MillConfig] = None,
    recipe: Optional[MillRecipe] = None,
    duration_s: float = 60.0,
    dt: float = 0.001,
    initial_holdup_kg: float = 0.0,
    device: str = "cpu",
) -> MillingResult:
    """Convenience function to run a milling simulation.

    Args:
        config: Mill configuration
        recipe: Operating recipe
        duration_s: Simulation duration [s]
        dt: Timestep [s]
        initial_holdup_kg: Initial mass in mill
        device: "cpu" or "cuda"

    Returns:
        MillingResult with time series and summary
    """
    sim = HammerMillSimulator.create(config=config, recipe=recipe, device=device)
    sim.initialize(initial_holdup_kg=initial_holdup_kg)
    return sim.run(duration_s=duration_s, dt=dt)
