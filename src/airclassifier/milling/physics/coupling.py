"""
Coupled Milling Engine
======================

Orchestrates the physics simulation step sequence for the hammer mill.
Combines transport, impact, breakage, and screen classification into
a coherent simulation loop.

Step sequence per timestep:
    1. FEED — Inject new particles from inlet
    2. TRANSPORT — Advect particles in chamber
    3. IMPACT — Detect hammer-particle collisions
    4. BREAKAGE — Apply size reduction to impacted particles
    5. SCREEN — Test passage through screen
    6. DISCHARGE — Move passed particles to outlet
    7. RECORD — Log state and KPIs

Requirements for milled flour D50 (discharge product)
----------------------------------------------------
To achieve discharge product with D50 in the target range (e.g. 23.7–31.1 µm
for yellow pea protein separation, NIH):

1. **Screen aperture (m)**  
   Only particles with size <= aperture [m] can pass. Recipe screen_aperture_mm
   is converted to meters (aperture_m) in the screen classifier. 0.75 mm → 0.00075 m.

2. **Breakage**  
   Seeds (e.g. feed_d50_um ~3000 µm) must be reduced by repeated impacts:
   - Particles must enter the hammer zone (transport + impact) to get impact_flags=1.
   - Selection and breakage function (BreakageParams) must produce fines;
     d_min_um allows particles down to 5 µm; fine-regime gamma/clamps control
     how small daughters get per breakage event.
   - Run long enough (or sufficient residence time) so that a significant mass
     reaches sizes well below the aperture (10–100 µm).

3. **Discharge D50 reported in state**  
   state.d10_m, state.d50_m, state.d90_m are the cumulative **discharge product**
   PSD (median size by mass of all particles that have passed the screen this run).
   They are in **meters**. So discharge D50 in µm = state.d50_m * 1e6.
   At start of run the discharge buffer is empty (cleared in initialize());
   as particles pass, the buffer accumulates and D50 converges toward steady state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .impact import ImpactSolver, ImpactStats
from .breakage import BreakageModel, BreakageStats
from .screen_classifier import ScreenClassifier, ScreenStats
from .convergence import ConvergenceDetector, TerminationConfig
from ..kernels import transport_step_np, transport_step_warp, GRAVITY, WARP_AVAILABLE
from ..config import MillConfig, MillRecipe, BreakageParams, ScreenConfig

from typing import TYPE_CHECKING
if WARP_AVAILABLE:
    import warp as wp
if TYPE_CHECKING:
    from airclassifier.pretreatment.physics.coupling import OutletState


@dataclass
class MillingStepState:
    """State snapshot at each timestep."""

    time_s: float = 0.0
    rotor_theta_rad: float = 0.0

    # Particle counts
    num_particles: int = 0
    num_fed: int = 0
    num_discharged: int = 0

    # Masses
    holdup_kg: float = 0.0
    feed_rate_kg_per_s: float = 0.0
    discharge_rate_kg_per_s: float = 0.0

    # Impact stats
    num_impacts: int = 0
    total_impact_energy_j: float = 0.0
    mean_impact_energy_j: float = 0.0

    # Breakage stats
    num_breakage_events: int = 0
    mean_size_reduction: float = 1.0
    num_fragments_created: int = 0

    # Screen stats
    num_passed_screen: int = 0
    screen_passage_rate: float = 0.0

    # PSD (d10, d50, d90 in meters)
    d10_m: float = 0.0
    d50_m: float = 0.0
    d90_m: float = 0.0

    # Power
    power_kw: float = 0.0


@dataclass
class ParticleState:
    """State of all particles in the mill."""

    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    velocities: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    sizes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    masses: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    residence_times: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    # Per-particle breakage count (number of times this particle has been broken)
    break_count: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))

    @property
    def count(self) -> int:
        return len(self.positions)

    @property
    def total_mass(self) -> float:
        return float(self.masses.sum())


@dataclass
class DischargeParticle:
    """Particle flowing through the discharge chute (for visualization)."""
    position: np.ndarray  # [3] x, y, z
    velocity: np.ndarray  # [3] vx, vy, vz
    size: float
    age: float  # Time since discharge [s]


@dataclass
class CoupledMillingEngine:
    """Orchestrates the hammer mill physics simulation.

    Combines all physics solvers and manages the particle state.
    """

    # Configuration
    config: MillConfig = field(default_factory=MillConfig)
    recipe: MillRecipe = field(default_factory=MillRecipe)

    # Physics solvers
    impact_solver: ImpactSolver = None
    breakage_model: BreakageModel = None
    screen_classifier: ScreenClassifier = None

    # Particle state
    particles: ParticleState = field(default_factory=ParticleState)

    # Time tracking
    time_s: float = 0.0
    rotor_theta: float = 0.0

    # Chamber geometry
    chamber_radius: float = 0.22
    chamber_length: float = 0.40

    # Feed state
    _feed_rate_kg_per_s: float = 0.0
    _feed_particle_size: float = 0.003  # 3mm for whole seeds
    _feed_particle_mass: float = 0.00015  # Mass per particle [kg] (internal; ~150 mg per pea)
    _seeds_feed_mass_kg: float = 0.0     # Total seeds mass to feed [kg]; 0 = unlimited
    _total_fed_mass_kg: float = 0.0  # Cumulative mass fed this run (for batch cap)
    _feed_accumulator: float = 0.0
    _inlet_temperature_c: float = 25.0   # Passthrough from pretreatment
    _inlet_moisture_wb: float = 0.12     # Passthrough from pretreatment

    # Discharge tracking
    _last_discharge_mass: float = 0.0
    _last_feed_mass: float = 0.0

    # Discharge particle visualization (particles flowing through outlet)
    _discharge_particles: List[DischargeParticle] = field(default_factory=list)
    _discharge_max_age: float = 0.5  # Remove after 0.5s (fallen through chute)
    _discharge_chute_y: float = -0.25  # Y position of discharge outlet
    _discharge_chute_z: float = -0.15  # Z position (below screen)
    _max_discharge_vis: int = 200  # Max discharge particles to visualize

    # History
    history: List[MillingStepState] = field(default_factory=list)

    # Convergence detection
    convergence_detector: Optional[ConvergenceDetector] = None
    termination_config: Optional[TerminationConfig] = None

    # Device
    device: str = "cpu"

    # Warp arrays for transport (when device=cuda); resized when particle count changes
    _wp_positions: Optional[Any] = None
    _wp_velocities: Optional[Any] = None
    _wp_sizes: Optional[Any] = None
    _wp_masses: Optional[Any] = None
    _wp_residence_times: Optional[Any] = None
    _wp_n: int = 0

    def __post_init__(self):
        """Initialize physics solvers."""
        if self.impact_solver is None:
            self.impact_solver = ImpactSolver.from_config(self.config, self.device)

        if self.breakage_model is None:
            self.breakage_model = BreakageModel(
                params=BreakageParams(),
                device=self.device,
            )

        if self.screen_classifier is None:
            self.screen_classifier = ScreenClassifier.from_config(
                self.config,
                device=self.device,
            )

        # Update chamber geometry from config
        self.chamber_radius = self.config.housing_inner_radius_m
        self.chamber_length = self.config.housing_length_m

    def load_recipe(self, recipe: MillRecipe):
        """Load a milling recipe.

        Args:
            recipe: Milling recipe with operating parameters
        """
        self.recipe = recipe
        self._feed_rate_kg_per_s = recipe.feed_rate_kg_per_hr / 3600.0
        self._feed_particle_size = recipe.feed_d50_um * 1e-6
        self._feed_particle_mass = recipe.feed_particle_mass_kg
        self._seeds_feed_mass_kg = recipe.seeds_feed_mass_kg
        self._inlet_temperature_c = recipe.feed_temperature_c
        self._inlet_moisture_wb = recipe.feed_moisture_wb

        # Update screen aperture
        self.screen_classifier.config.aperture_mm = recipe.screen_aperture_mm

    def set_inlet_state(self, outlet_state: "OutletState"):
        """Set feed from pretreatment outlet (pipeline integration).

        Args:
            outlet_state: OutletState from airclassifier.pretreatment (GP-15 outfeed).
        """
        self._feed_rate_kg_per_s = outlet_state.throughput_kg_per_hr / 3600.0
        self._inlet_temperature_c = outlet_state.avg_temperature_c
        self._inlet_moisture_wb = outlet_state.avg_moisture_wb
        if self.recipe is not None:
            self.recipe.feed_temperature_c = outlet_state.avg_temperature_c
            self.recipe.feed_moisture_wb = outlet_state.avg_moisture_wb

    def initialize(
        self,
        initial_particles: int = 0,
        initial_holdup_kg: float = 0.0,
    ):
        """Initialize the simulation.

        Args:
            initial_particles: Number of particles to start with
            initial_holdup_kg: Initial mass in mill (alternative to particle count)
        """
        self.time_s = 0.0
        self.rotor_theta = 0.0
        self.history.clear()
        self.screen_classifier.clear_discharge_buffer()
        self._feed_accumulator = 0.0
        self._total_fed_mass_kg = 0.0
        self._last_discharge_mass = 0.0
        self._last_feed_mass = 0.0
        self._discharge_particles.clear()  # Clear discharge visualization

        # Reset convergence detector if configured
        if self.convergence_detector is not None:
            self.convergence_detector.reset()

        # Initialize particles
        if initial_holdup_kg > 0:
            # Compute particle count from mass
            initial_particles = int(initial_holdup_kg / self._feed_particle_mass)

        if initial_particles > 0:
            self._create_initial_particles(initial_particles)
        else:
            self.particles = ParticleState()

    def _create_initial_particles(self, n: int):
        """Create initial particle distribution in chamber."""
        # Random positions within chamber
        rng = np.random.default_rng()

        # Distribute along chamber length
        x = rng.uniform(0.05, self.chamber_length - 0.05, n)

        # Random radial positions (uniform in disk)
        r = self.chamber_radius * 0.8 * np.sqrt(rng.uniform(0, 1, n))
        theta = rng.uniform(0, 2 * np.pi, n)
        y = r * np.cos(theta)
        z = r * np.sin(theta)

        positions = np.column_stack([x, y, z]).astype(np.float32)

        # Random velocities (low initial speed)
        velocities = rng.uniform(-0.5, 0.5, (n, 3)).astype(np.float32)

        # Feed particle size with some variation
        sizes = (self._feed_particle_size * rng.uniform(0.8, 1.2, n)).astype(np.float32)

        # Masses (proportional to size^3)
        size_ratio = sizes / self._feed_particle_size
        masses = (self._feed_particle_mass * size_ratio ** 3).astype(np.float32)

        # Zero residence time
        residence_times = np.zeros(n, dtype=np.float32)

        break_count = np.zeros(n, dtype=np.int32)
        self.particles = ParticleState(
            positions=positions,
            velocities=velocities,
            sizes=sizes,
            masses=masses,
            residence_times=residence_times,
            break_count=break_count,
        )

    def step(self, dt: float) -> MillingStepState:
        """Advance the simulation by one timestep.

        Args:
            dt: Timestep [s]

        Returns:
            State at end of step
        """
        # Update rotor angle
        omega = self.recipe.rotor_omega
        self.rotor_theta += omega * dt
        self.rotor_theta = self.rotor_theta % (2 * np.pi)

        # Update impact solver rotor state
        self.impact_solver.set_rotor_state(self.rotor_theta, omega)

        # Store pre-step state for rate calculations
        pre_holdup = self.particles.total_mass

        # --- 1. FEED ---
        num_fed = self._feed_step(dt)

        # --- 2. TRANSPORT ---
        self._transport_step(dt)

        # --- 3. IMPACT ---
        impact_flags, impact_energies = self._impact_step(dt)

        # --- 4. BREAKAGE ---
        break_flags = self._breakage_step(impact_flags, impact_energies)

        # --- 5. SCREEN ---
        num_discharged = self._screen_step()

        # --- 6. DISCHARGE ---
        # (handled in screen step)

        # --- 7. RECORD ---
        self.time_s += dt
        post_holdup = self.particles.total_mass

        # Compute rates
        discharge_mass = pre_holdup + (num_fed * self._feed_particle_mass) - post_holdup
        feed_mass = num_fed * self._feed_particle_mass

        # Get PSD stats (cumulative discharge product; sizes in meters)
        d10, d50, d90 = self.screen_classifier.get_d_values()
        aperture_m = self.screen_classifier.config.aperture_m
        # Sanity: discharge D50 cannot exceed screen aperture (only size <= aperture can pass)
        if d50 > 0 and aperture_m > 0 and d50 > aperture_m * 1.01:
            import warnings
            warnings.warn(
                f"Milling: discharge d50_m ({d50:.2e} m) > aperture_m ({aperture_m:.2e} m); "
                "check units (aperture must be in meters)."
            )

        # Compute power
        impact_power = self.impact_solver.compute_power_draw(dt)
        no_load_power = self.config.no_load_power_kw * 1000  # W
        total_power = (impact_power + no_load_power) / 1000  # kW

        discharge_rate = discharge_mass / dt if dt > 0 else 0.0

        state = MillingStepState(
            time_s=self.time_s,
            rotor_theta_rad=self.rotor_theta,
            num_particles=self.particles.count,
            num_fed=num_fed,
            num_discharged=num_discharged,
            holdup_kg=post_holdup,
            feed_rate_kg_per_s=feed_mass / dt if dt > 0 else 0.0,
            discharge_rate_kg_per_s=discharge_rate,
            num_impacts=self.impact_solver.stats.num_impacts,
            total_impact_energy_j=self.impact_solver.stats.total_impact_energy,
            mean_impact_energy_j=self.impact_solver.stats.mean_impact_energy,
            num_breakage_events=self.breakage_model.stats.num_breakage_events,
            mean_size_reduction=self.breakage_model.stats.size_reduction_ratio,
            num_fragments_created=self.breakage_model.stats.num_fragments_created,
            num_passed_screen=self.screen_classifier.stats.num_passed,
            screen_passage_rate=self.screen_classifier.stats.passage_rate,
            d10_m=d10,
            d50_m=d50,
            d90_m=d90,
            power_kw=total_power,
        )

        self.history.append(state)

        # Update convergence detector if configured
        if self.convergence_detector is not None:
            self.convergence_detector.update(
                time_s=self.time_s,
                d50_m=d50,
                discharge_rate_kg_per_s=discharge_rate,
                power_kw=total_power,
                dt=dt,
                particle_count=self.particles.count,
            )

        # Update discharge visualization particles
        self.update_discharge_visualization(dt)

        return state

    def set_termination_config(self, config: TerminationConfig) -> None:
        """Set termination configuration and create detector.

        Args:
            config: Termination configuration
        """
        self.termination_config = config
        self.convergence_detector = config.create_detector()

    def check_termination(self) -> Tuple[bool, str]:
        """Check if simulation should terminate based on physics criteria.

        Returns:
            (should_stop, reason) tuple
        """
        if self.convergence_detector is None:
            return False, ""
        return self.convergence_detector.should_terminate()

    def get_convergence_progress(self) -> float:
        """Get progress percentage for physics-based termination modes.

        Returns:
            Progress as percentage (0-100), or 0 if not applicable.
        """
        if self.convergence_detector is not None:
            return self.convergence_detector.progress_pct
        return 0.0

    def _feed_step(self, dt: float) -> int:
        """Inject new particles from feed.

        Returns:
            Number of particles fed
        """
        # When seeds feed mass is set, do not exceed it (batch mode)
        if self._seeds_feed_mass_kg > 0 and self._total_fed_mass_kg >= self._seeds_feed_mass_kg:
            return 0

        # Accumulate feed mass
        self._feed_accumulator += self._feed_rate_kg_per_s * dt

        # Convert to particles
        num_new = int(self._feed_accumulator / self._feed_particle_mass)
        if num_new <= 0:
            return 0

        # Cap by remaining seeds mass when configured
        if self._seeds_feed_mass_kg > 0:
            remaining_kg = self._seeds_feed_mass_kg - self._total_fed_mass_kg
            max_new = int(remaining_kg / self._feed_particle_mass)
            num_new = min(num_new, max_new)
            if num_new <= 0:
                return 0

        self._feed_accumulator -= num_new * self._feed_particle_mass
        self._total_fed_mass_kg += num_new * self._feed_particle_mass

        # Create new particles at feed inlet
        rng = np.random.default_rng()

        # Position at feed chute outlet (top of housing)
        feed_x = 0.15  # Center of feed opening
        feed_y = self.chamber_radius * 0.9  # Near top
        feed_z = 0.0

        x = rng.normal(feed_x, 0.02, num_new)
        y = np.full(num_new, feed_y)
        z = rng.normal(feed_z, 0.03, num_new)

        new_pos = np.column_stack([x, y, z]).astype(np.float32)

        # Initial velocity (falling into chamber)
        new_vel = np.zeros((num_new, 3), dtype=np.float32)
        new_vel[:, 1] = -1.0  # Falling

        # Sizes
        new_sizes = (self._feed_particle_size * rng.uniform(0.8, 1.2, num_new)).astype(np.float32)

        # Masses
        size_ratio = new_sizes / self._feed_particle_size
        new_masses = (self._feed_particle_mass * size_ratio ** 3).astype(np.float32)

        # Residence times
        new_res = np.zeros(num_new, dtype=np.float32)

        # Append to existing particles
        new_break = np.zeros(num_new, dtype=np.int32)
        self.particles = ParticleState(
            positions=np.vstack([self.particles.positions, new_pos]) if self.particles.count > 0 else new_pos,
            velocities=np.vstack([self.particles.velocities, new_vel]) if self.particles.count > 0 else new_vel,
            sizes=np.concatenate([self.particles.sizes, new_sizes]) if self.particles.count > 0 else new_sizes,
            masses=np.concatenate([self.particles.masses, new_masses]) if self.particles.count > 0 else new_masses,
            residence_times=np.concatenate([self.particles.residence_times, new_res]) if self.particles.count > 0 else new_res,
            break_count=np.concatenate([self.particles.break_count, new_break]) if self.particles.count > 0 else new_break,
        )

        return num_new

    def _transport_step(self, dt: float):
        """Advect particles in chamber. Uses Warp GPU kernel when device=cuda."""
        if self.particles.count == 0:
            return

        use_warp = (
            self.device == "cuda"
            and WARP_AVAILABLE
            and transport_step_warp is not None
        )
        if use_warp:
            n = self.particles.count
            if self._wp_n != n or self._wp_positions is None:
                self._wp_n = n
                self._wp_positions = wp.array(
                    self.particles.positions, dtype=wp.vec3, device="cuda"
                )
                self._wp_velocities = wp.array(
                    self.particles.velocities, dtype=wp.vec3, device="cuda"
                )
                self._wp_sizes = wp.array(
                    self.particles.sizes, dtype=float, device="cuda"
                )
                self._wp_masses = wp.array(
                    self.particles.masses, dtype=float, device="cuda"
                )
                self._wp_residence_times = wp.array(
                    self.particles.residence_times, dtype=float, device="cuda"
                )
            else:
                wp.copy(self._wp_positions, wp.array(self.particles.positions, dtype=wp.vec3, device="cuda"))
                wp.copy(self._wp_velocities, wp.array(self.particles.velocities, dtype=wp.vec3, device="cuda"))
                wp.copy(self._wp_sizes, wp.array(self.particles.sizes, dtype=float, device="cuda"))
                wp.copy(self._wp_masses, wp.array(self.particles.masses, dtype=float, device="cuda"))
                wp.copy(self._wp_residence_times, wp.array(self.particles.residence_times, dtype=float, device="cuda"))

            transport_step_warp(
                positions=self._wp_positions,
                velocities=self._wp_velocities,
                sizes=self._wp_sizes,
                masses=self._wp_masses,
                residence_times=self._wp_residence_times,
                chamber_radius=self.chamber_radius,
                chamber_length=self.chamber_length,
                rotor_omega=self.recipe.rotor_omega,
                dt=dt,
            )

            self.particles.positions = self._wp_positions.numpy()
            self.particles.velocities = self._wp_velocities.numpy()
            self.particles.residence_times = self._wp_residence_times.numpy()
        else:
            new_pos, new_vel, new_res = transport_step_np(
                positions=self.particles.positions,
                velocities=self.particles.velocities,
                sizes=self.particles.sizes,
                masses=self.particles.masses,
                residence_times=self.particles.residence_times,
                chamber_radius=self.chamber_radius,
                chamber_length=self.chamber_length,
                rotor_omega=self.recipe.rotor_omega,
                dt=dt,
            )
            self.particles.positions = new_pos
            self.particles.velocities = new_vel
            self.particles.residence_times = new_res

    def _impact_step(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Detect and resolve impacts.

        Returns:
            (impact_flags, impact_energies)
        """
        if self.particles.count == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        impact_flags, impact_energies, new_vel = self.impact_solver.step(
            positions=self.particles.positions,
            velocities=self.particles.velocities,
            sizes=self.particles.sizes,
            masses=self.particles.masses,
            dt=dt,
        )

        self.particles.velocities = new_vel
        return impact_flags, impact_energies

    def _breakage_step(
        self,
        impact_flags: np.ndarray,
        impact_energies: np.ndarray,
    ) -> np.ndarray:
        """Apply breakage to impacted particles.

        If multi-fragment breakage is enabled, secondary fragments are generated
        and appended to the particle arrays (up to max_particle_count).

        Returns:
            break_flags
        """
        if self.particles.count == 0:
            return np.array([], dtype=np.int32)

        (
            new_sizes, new_masses, break_flags,
            frag_sizes, frag_masses, parent_indices, num_frags,
        ) = self.breakage_model.step_lagrangian(
            sizes=self.particles.sizes,
            masses=self.particles.masses,
            impact_flags=impact_flags,
            impact_energies=impact_energies,
        )

        self.particles.sizes = new_sizes
        self.particles.masses = new_masses
        # Track per-particle breakage count
        self.particles.break_count = self.particles.break_count.astype(np.int32) + break_flags.astype(np.int32)

        # --- Append secondary fragments ---
        if frag_sizes is not None and num_frags > 0:
            max_cap = self.breakage_model.params.max_particle_count
            capacity_left = max_cap - self.particles.count

            if capacity_left > 0:
                n_add = min(num_frags, capacity_left)

                # If we must truncate, redistribute dropped fragment mass to primaries
                if n_add < num_frags:
                    dropped_mass = frag_masses[n_add:].sum()
                    # Distribute dropped mass back to their parent primaries
                    for j in range(n_add, num_frags):
                        pidx = parent_indices[j]
                        self.particles.masses[pidx] += frag_masses[j]

                frag_sizes = frag_sizes[:n_add]
                frag_masses = frag_masses[:n_add]
                parent_indices = parent_indices[:n_add]

                # Fragment positions/velocities: copy from parent + noise
                rng = np.random.default_rng()
                p_cfg = self.breakage_model.params
                pos_noise = p_cfg.fragment_position_noise_m
                vel_noise = p_cfg.fragment_velocity_noise_m_per_s

                frag_pos = self.particles.positions[parent_indices].copy()
                frag_pos += rng.normal(0, pos_noise, frag_pos.shape).astype(np.float32)

                frag_vel = self.particles.velocities[parent_indices].copy()
                frag_vel += rng.normal(0, vel_noise, frag_vel.shape).astype(np.float32)

                frag_res = np.zeros(n_add, dtype=np.float32)
                frag_break = np.ones(n_add, dtype=np.int32)  # born from breakage

                # Append to particle state
                self.particles = ParticleState(
                    positions=np.vstack([self.particles.positions, frag_pos]),
                    velocities=np.vstack([self.particles.velocities, frag_vel]),
                    sizes=np.concatenate([self.particles.sizes, frag_sizes.astype(np.float32)]),
                    masses=np.concatenate([self.particles.masses, frag_masses.astype(np.float32)]),
                    residence_times=np.concatenate([self.particles.residence_times, frag_res]),
                    break_count=np.concatenate([self.particles.break_count, frag_break]),
                )

        return break_flags

    def _screen_step(self) -> int:
        """Test screen passage and discharge.

        Returns:
            Number of particles discharged
        """
        if self.particles.count == 0:
            return 0

        # Get original positions/sizes before filtering for discharge viz
        orig_positions = self.particles.positions.copy()
        orig_sizes = self.particles.sizes.copy()

        ret_pos, ret_vel, ret_sizes, ret_masses, passage_flags = self.screen_classifier.step(
            positions=self.particles.positions,
            velocities=self.particles.velocities,
            sizes=self.particles.sizes,
            masses=self.particles.masses,
        )

        num_discharged = int(passage_flags.sum())

        # Create discharge visualization particles for passed particles
        # Particles exit from their actual screen-surface position (digital twin realism)
        if num_discharged > 0:
            passed_mask = passage_flags == 1
            passed_pos = orig_positions[passed_mask]
            passed_sizes = orig_sizes[passed_mask]

            rng = np.random.default_rng()
            capacity = self._max_discharge_vis - len(self._discharge_particles)
            for i in range(min(num_discharged, capacity)):
                if i >= len(passed_pos):
                    break

                pos = passed_pos[i]
                x = pos[0]

                # Start at actual screen position, shifted just outside the screen
                r_yz = math.sqrt(pos[1] ** 2 + pos[2] ** 2)
                if r_yz > 0.01:
                    # Push 3cm past screen surface (particle exits through screen holes)
                    exit_scale = (r_yz + 0.03) / r_yz
                    y = pos[1] * exit_scale
                    z = pos[2] * exit_scale
                else:
                    y = self._discharge_chute_y
                    z = self._discharge_chute_z

                # Velocity: radially outward from screen + gravity bias
                r_hat_y = pos[1] / max(r_yz, 0.01)
                r_hat_z = pos[2] / max(r_yz, 0.01)
                vel = np.array([
                    rng.uniform(-0.1, 0.1),
                    r_hat_y * 0.5 + rng.uniform(-0.3, 0.0),
                    r_hat_z * 0.5 + rng.uniform(-0.1, 0.1),
                ], dtype=np.float32)

                size = float(passed_sizes[i])

                self._discharge_particles.append(DischargeParticle(
                    position=np.array([x, y, z], dtype=np.float32),
                    velocity=vel,
                    size=size,
                    age=0.0,
                ))

        # Update particle state to retained only
        retained_mask = passage_flags == 0
        self.particles = ParticleState(
            positions=ret_pos,
            velocities=ret_vel,
            sizes=ret_sizes,
            masses=ret_masses,
            residence_times=self.particles.residence_times[retained_mask],
            break_count=self.particles.break_count[retained_mask],
        )

        return num_discharged

    def run(
        self,
        duration_s: float,
        dt: float = 0.001,
    ) -> List[MillingStepState]:
        """Run simulation for a duration.

        Args:
            duration_s: Total duration [s]
            dt: Timestep [s]

        Returns:
            List of step states
        """
        num_steps = int(duration_s / dt)
        states = []

        for _ in range(num_steps):
            state = self.step(dt)
            states.append(state)

        return states

    def get_discharge_psd(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Get PSD of discharged material.

        Returns:
            (size_classes, mass_fractions, total_mass)
        """
        size_classes = self.breakage_model.get_size_classes()
        mass_fractions, total_mass = self.screen_classifier.get_discharge_psd(size_classes)
        return size_classes, mass_fractions, total_mass

    def get_retained_psd(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Get PSD of particles still retained in the mill chamber.

        Returns:
            (size_classes, mass_fractions, total_mass)
        """
        size_classes = self.breakage_model.get_size_classes()
        sizes = self.particles.sizes
        masses = self.particles.masses
        total_mass = float(masses.sum()) if len(masses) > 0 else 0.0

        n = len(size_classes) - 1
        mass_fractions = np.zeros(n)

        if len(sizes) > 0 and total_mass > 0:
            for i in range(n):
                mask = (sizes >= size_classes[i]) & (sizes < size_classes[i + 1])
                mass_fractions[i] = masses[mask].sum()
            mass_fractions /= total_mass

        return size_classes, mass_fractions, total_mass

    def update_discharge_visualization(self, dt: float) -> None:
        """Update discharge particle positions for visualization.

        Applies gravity, updates positions, and removes aged particles.

        Args:
            dt: Timestep [s]
        """
        gravity_y = -9.81  # m/s^2

        # Update each discharge particle
        new_particles = []
        for p in self._discharge_particles:
            # Update velocity (gravity)
            p.velocity[1] += gravity_y * dt

            # Update position
            p.position += p.velocity * dt

            # Update age
            p.age += dt

            # Keep if not too old and not fallen too far
            if p.age < self._discharge_max_age and p.position[1] > -0.6:
                new_particles.append(p)

        self._discharge_particles = new_particles

    def get_all_visible_particles(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get positions and sizes of all visible particles.

        Combines chamber particles with discharge visualization particles.

        Returns:
            (positions [n, 3], sizes [n]) including both chamber and discharge
        """
        # Start with chamber particles
        if self.particles.count > 0:
            chamber_pos = self.particles.positions
            chamber_sizes = self.particles.sizes
        else:
            chamber_pos = np.zeros((0, 3), dtype=np.float32)
            chamber_sizes = np.zeros(0, dtype=np.float32)

        # Add discharge particles
        if self._discharge_particles:
            discharge_pos = np.array([p.position for p in self._discharge_particles], dtype=np.float32)
            discharge_sizes = np.array([p.size for p in self._discharge_particles], dtype=np.float32)

            all_pos = np.vstack([chamber_pos, discharge_pos]) if chamber_pos.size > 0 else discharge_pos
            all_sizes = np.concatenate([chamber_sizes, discharge_sizes]) if chamber_sizes.size > 0 else discharge_sizes
        else:
            all_pos = chamber_pos
            all_sizes = chamber_sizes

        return all_pos, all_sizes

    def clear_discharge_visualization(self) -> None:
        """Clear all discharge visualization particles."""
        self._discharge_particles.clear()
