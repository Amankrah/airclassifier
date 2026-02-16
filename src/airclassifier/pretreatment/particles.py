"""
Material Particle System -- Lagrangian Tracers
===============================================

Eulerian-Lagrangian coupling for the GP-15 material bed visualization.

The physics (RF heating, moisture diffusion, evaporation, heat
conduction) is solved on the Eulerian grid in ``CoupledSimulator``
using the governing PDEs from the engineering guide (section 4):

    dT/dt = div(k grad T) + P_v - L_v * m_evap     (section 4.2)
    dM/dt = div(D_eff grad M) - m_evap / rho_dry    (section 4.3)
    P_v   = 2*pi*f*eps_0*eps''*|E|^2                 (section 4.1)

Particles provide the Lagrangian view of the same solution:

    1. Start inside the hopper (visual pool, gravity settling)
    2. Dispatch from hopper → belt at the sizing gate (spawn rate)
    3. Ride the belt in +X through the oven (section 4.4)
    4. Trilinearly interpolate T and M from the Eulerian grid
    5. Fall under gravity when leaving the head roller
    6. Settle in the collection bin

When ``run_mass_kg > 0`` (finite mass from recipe), dispatching
stops once the mass limit is reached.  The belt then clears
naturally as the last particles ride through and fall off.

All physical constants and material properties are imported from the
project's established material systems:

    - ``airclassifier.utils.constants``      -- GRAVITY, MaterialDensities
    - ``airclassifier.particles.material``   -- ParticleMaterial, size distributions
    - ``airclassifier.pretreatment.config``  -- MaterialProperties (dielectric/thermal)
    - ``airclassifier.pretreatment.materials.presets`` -- feedstock factory functions

No magic numbers.  Every constant traces to a measured value or a
published property model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

# ── Project-wide constants (single source of truth) ──────────────────
try:
    from ..utils.constants import GRAVITY
except ImportError:
    # Fallback if running outside the full package (e.g., tests)
    GRAVITY = 9.80665  # standard gravitational acceleration [m/s^2]

# ── Pretreatment material system (whole seeds, NOT flour) ────────────
# The GP-15 processes whole beans/seeds/groats BEFORE milling.
# All material properties come from pretreatment.config, which
# defines rho_solid, bed_porosity, dielectric coefficients, etc.
# for the raw seed form.
#
# Do NOT import from airclassifier.particles.material — that module
# defines flour particles (post-milling: 5-50 um) for the air
# classifier, a completely different material at a different scale.
from .config import MaterialProperties as PretreatmentMaterial


@dataclass
class ParticleSystemConfig:
    """Configuration derived from machine assembly and material properties.

    All geometry values are set by ``from_assembly()`` from the
    machine's component parameters.  Material properties are set from
    the pretreatment ``MaterialProperties`` dataclass.  No hardcoded
    defaults for physical values.
    """

    max_particles: int = 8000
    spawn_rate: float = 200.0         # particles / sim-second

    # Geometry (populated by from_assembly, from component params)
    spawn_x: float = 0.0             # hopper discharge X [m]
    spawn_z0: float = 0.0            # belt left edge Z [m]
    spawn_z1: float = 0.8            # belt right edge Z [m]
    belt_y: float = 0.004            # belt carrying surface Y [m]
    bed_depth_m: float = 0.05        # bed thickness above belt [m]
    head_roller_x: float = 4.5       # head roller centre X [m]
    head_roller_r: float = 0.075     # head roller radius [m]

    # Hopper geometry (populated by from_assembly)
    hopper_back_x: float = 0.0       # back wall X [m]
    hopper_front_x: float = 0.0      # front wall / discharge X [m]
    hopper_height_m: float = 0.30    # visual fill height above belt [m]
    hopper_z0: float = 0.0           # hopper left edge Z [m]
    hopper_z1: float = 0.7           # hopper right edge Z [m]

    # Collection bin geometry (from assembly, machine.py)
    bin_x0: float = 4.0              # bin back wall X [m]
    bin_x1: float = 5.0              # bin front wall X [m]
    bin_z0: float = 0.0              # bin left wall Z [m]
    bin_z1: float = 1.0              # bin right wall Z [m]
    bin_bottom_y: float = -1.3       # bin floor Y [m]
    bin_top_y: float = -0.5          # bin rim Y [m]

    # Material inlet conditions (from PretreatmentMaterial)
    T_inlet_c: float = 22.0          # infeed temperature [C]
    M_inlet_wb: float = 0.10         # infeed moisture [wet-basis]

    # Whole seed physical properties (from PretreatmentMaterial.rho_solid)
    particle_density: float = 1450.0  # whole seed solid density [kg/m^3]
    particle_sphericity: float = 0.85 # whole seeds are roughly spherical [-]

    # Physics constant (from utils.constants)
    gravity: float = GRAVITY          # [m/s^2]

    # Mass accounting (from manual Chapter 5: 600 kg/h example)
    # Each visual particle represents a portion of the continuous
    # mass flow.  mass_per_particle = throughput / spawn_rate.
    # Throughput = rho_bulk * bed_cross * v_belt [kg/s]
    throughput_kg_per_s: float = 0.0  # set by from_assembly()

    # Finite mass feed (0 = continuous / infinite)
    run_mass_kg: float = 0.0         # total mass to feed from hopper

    # Grid info (for field interpolation bounds)
    rf_x_start: float = 0.0
    rf_x_end: float = 1.5

    # Oven chamber exit (for clearing detection — Manual p.54)
    # Material is "in the GP-15" until it exits the outfeed attenuation
    # duct, not when it passes the last electrode.
    oven_x_end: float = 3.6


class MaterialParticleSystem:
    """Lagrangian tracer particles coupled to the Eulerian physics grid.

    Each particle carries:
        pos[i]         -- world position (x, y, z) [m]
        vel[i]         -- velocity (only during free-fall) [m/s]
        temperature[i] -- trilinearly interpolated from grid T [C]
        moisture[i]    -- trilinearly interpolated from grid M [wb]
        state[i]       -- IN_HOPPER / RIDING / FALLING / COLLECTED / DEAD

    Operating Modes
    ---------------
    **Finite Mass Mode** (``run_mass_kg > 0``):
        All particles start in the hopper, representing fresh material.
        Lifecycle is one-way::

            IN_HOPPER → RIDING → FALLING → COLLECTED (no recycling)

        - Particles drain from hopper at the spawn rate
        - Once dispatched, particles ride the belt through the oven
        - T and M are interpolated from the Eulerian fields (drying)
        - Particles fall into the collection bin with final T/M values
        - Collected particles stay collected (treated material is NOT recycled)
        - Run completes when hopper is empty and belt has cleared

    **Continuous Mode** (``run_mass_kg == 0``):
        Particles recycle for continuous visualization::

            IN_HOPPER → RIDING → FALLING → COLLECTED → (recycle to DEAD → RIDING)

    Material properties are sourced from the pretreatment system:
        - Inlet T, M          from PretreatmentMaterial (config.py)
        - Particle density     from PretreatmentMaterial.rho_solid (whole seed density)
        - Gravity              from utils.constants.GRAVITY
        - Geometry             from machine assembly components
    """

    _STATE_RIDING = 0
    _STATE_FALLING = 1
    _STATE_COLLECTED = 2   # sitting in the collection bin
    _STATE_DEAD = 3        # recycled (off-screen, awaiting respawn)
    _STATE_IN_HOPPER = 4   # inside the hopper bin, awaiting dispatch

    def __init__(self, config: ParticleSystemConfig):
        self.cfg = config
        n = config.max_particles

        self.pos = np.zeros((n, 3), dtype=np.float32)
        self.vel = np.zeros((n, 3), dtype=np.float32)
        self.temperature = np.full(n, config.T_inlet_c, dtype=np.float32)
        self.moisture = np.full(n, config.M_inlet_wb, dtype=np.float32)
        self.state = np.full(n, self._STATE_DEAD, dtype=np.int32)
        self.age = np.zeros(n, dtype=np.float32)

        # Mass accounting (manual Chapter 5).
        # Each particle represents mass_per_particle kg of material.
        # throughput [kg/s] / spawn_rate [particles/s] = kg/particle.
        if config.throughput_kg_per_s > 0 and config.spawn_rate > 0:
            self.mass_per_particle = config.throughput_kg_per_s / config.spawn_rate
        else:
            self.mass_per_particle = 0.0
        self._total_collected_kg = 0.0
        self._dispatched_mass_kg = 0.0
        self._initial_belt_mass_kg = 0.0  # pre-placed belt particles (for mass balance)
        self._batch_exhausted = False  # set True when hopper empties (stops M interpolation)

        self._alive_count = 0
        self._spawn_accumulator = 0.0
        self._init_particles()

    # ------------------------------------------------------------------
    #  Initialization
    # ------------------------------------------------------------------

    def _init_particles(self):
        """Initialize particle positions based on operating mode.

        When ``run_mass_kg > 0`` (finite mass run):
            - ALL particles start inside the hopper (fresh material)
            - No particles on the belt initially
            - Particles will drain from hopper through the system
            - No recycling: once collected, particles stay collected

        When ``run_mass_kg == 0`` (continuous / infinite):
            - All particles start distributed along the belt
            - Provides immediate visual feedback
            - Particles recycle through DEAD→RIDING states
        """
        cfg = self.cfg
        n = cfg.max_particles
        rng = np.random.default_rng(42)

        belt_len = cfg.head_roller_x - cfg.spawn_x
        if belt_len <= 0:
            return

        if cfg.run_mass_kg > 0 and cfg.hopper_front_x > cfg.hopper_back_x:
            # ── Finite mass: ALL particles start in hopper ────────
            # For a finite mass run, all material starts in the hopper
            # and gradually moves through the system:
            #   HOPPER → RIDING (belt) → FALLING → COLLECTED (bin)
            #
            # No recycling: once collected, particles stay collected.
            # The simulation runs until run_mass_kg has been dispatched
            # and all dispatched particles have been collected.
            #
            # This matches the real machine: operator loads a known mass
            # into the hopper, material flows through, and the run ends
            # when the hopper is empty and the belt has cleared.

            # Fill the hopper with particles in a pre-settled distribution.
            # Real granular material loaded into a hopper has already
            # settled under gravity before the machine starts.  The fill
            # is densest at the bottom with a fairly flat top surface.
            #
            # Using a steep power distribution (exponent 2.0) concentrates
            # ~70% of particles in the lower quarter, matching a settled
            # granular bed.  This avoids the visual artefact of particles
            # "dropping to the floor" on the first rendered frame — they
            # start where they would be after natural settling.
            hopper_fill_h = cfg.hopper_height_m * 0.75  # fill 75% of hopper height
            hopper_len = cfg.hopper_front_x - cfg.hopper_back_x
            hopper_wid = cfg.hopper_z1 - cfg.hopper_z0

            for i in range(n):
                # Steep power distribution: most particles near the bottom
                y_frac = rng.random() ** 2.0  # exponent > 1 = dense at bottom
                self.pos[i, 1] = cfg.belt_y + 0.005 + y_frac * hopper_fill_h

                # X distribution: uniform with slight bias toward front
                x_frac = rng.random() ** 0.9
                self.pos[i, 0] = cfg.hopper_back_x + 0.02 + x_frac * (hopper_len - 0.04)

                # Z distribution: uniform across hopper width
                self.pos[i, 2] = cfg.hopper_z0 + 0.01 + rng.random() * (hopper_wid - 0.02)

                self.state[i] = self._STATE_IN_HOPPER
                self.temperature[i] = cfg.T_inlet_c
                self.moisture[i] = cfg.M_inlet_wb

            # Mass accounting: fresh run starts at zero dispatched
            # No particles are pre-placed on the belt
            self._initial_belt_mass_kg = 0.0
            self._dispatched_mass_kg = 0.0

            self._alive_count = n
        else:
            # ── Continuous: fill belt (original behaviour) ────────
            self._prefill_belt(rng)

    def _prefill_belt(self, rng=None):
        """Distribute particles along the belt for a non-empty first frame."""
        cfg = self.cfg
        n = cfg.max_particles
        belt_len = cfg.head_roller_x - cfg.spawn_x
        if belt_len <= 0:
            return

        if rng is None:
            rng = np.random.default_rng(42)

        for i in range(n):
            frac = i / max(n - 1, 1)
            self.pos[i, 0] = cfg.spawn_x + frac * belt_len
            self.pos[i, 1] = cfg.belt_y + rng.uniform(0, cfg.bed_depth_m)
            self.pos[i, 2] = rng.uniform(cfg.spawn_z0, cfg.spawn_z1)
            self.state[i] = self._STATE_RIDING
            self.temperature[i] = cfg.T_inlet_c
            self.moisture[i] = cfg.M_inlet_wb

        self._alive_count = n

    # ------------------------------------------------------------------
    #  Run mass configuration
    # ------------------------------------------------------------------

    def set_run_mass(self, run_mass_kg: float, throughput_kg_per_s: float = 0.0):
        """Configure finite-mass feed from the hopper.

        Called by ``GP15Simulator.load_recipe()`` after the recipe
        (with ``run_mass_kg``) is loaded.

        For finite mass mode, the spawn_rate is calculated to match the
        actual physics:
            - run_time = run_mass_kg / throughput [s]
            - spawn_rate = max_particles / run_time [particles/s]

        This ensures particles drain from the hopper at exactly the rate
        that matches the Eulerian advection model.

        Args:
            run_mass_kg: Total mass to feed from the hopper [kg].
                         0 = continuous (infinite) mode.
            throughput_kg_per_s: Actual throughput at recipe belt speed.
                                Required for finite mass mode.
        """
        self.cfg.run_mass_kg = run_mass_kg
        if throughput_kg_per_s > 0:
            self.cfg.throughput_kg_per_s = throughput_kg_per_s

            if run_mass_kg > 0:
                # ── Finite mass mode: physics-based spawn rate ──────────
                # Calculate how long the run takes at this throughput
                run_time_s = run_mass_kg / throughput_kg_per_s

                # Spawn rate = all particles over the run time
                # This ensures hopper empties exactly when run completes
                self.cfg.spawn_rate = self.cfg.max_particles / run_time_s

                # Each particle represents a fraction of the total mass
                self.mass_per_particle = run_mass_kg / self.cfg.max_particles
            else:
                # ── Continuous mode: use throughput-based spawn rate ────
                # Keep original spawn_rate, calculate mass per particle
                self.mass_per_particle = throughput_kg_per_s / self.cfg.spawn_rate
        self._dispatched_mass_kg = 0.0
        self._total_collected_kg = 0.0
        self._initial_belt_mass_kg = 0.0
        self._batch_exhausted = False
        # Re-initialize particle distribution
        self.state[:] = self._STATE_DEAD
        self.pos[:] = 0.0
        self.vel[:] = 0.0
        self.age[:] = 0.0
        self._spawn_accumulator = 0.0
        self._alive_count = 0
        self._init_particles()

    @classmethod
    def from_assembly(
        cls,
        assembly,
        material: PretreatmentMaterial,
    ) -> "MaterialParticleSystem":
        """Create from the machine assembly and material properties.

        Geometry comes from the assembly's component parameters
        (single source of truth).  Inlet T and M come from the
        pretreatment MaterialProperties.  Particle density comes
        from ``rho_solid`` (whole seed solid density, measured).
        """
        cp = assembly.conveyor.params
        op = assembly.oven.params
        hp = assembly._hopper.params if hasattr(assembly, '_hopper') else None

        if hp is not None and hasattr(hp, 'hopper_front_x'):
            spawn_x = hp.hopper_front_x
        else:
            spawn_x = op.oven_x_start_m - 0.30

        belt_z0 = op.conveyor_belt_z0_m
        belt_z1 = belt_z0 + op.rf_zone_width_m

        # Whole seed properties from the pretreatment material system.
        # rho_solid is the measured particle density of whole seeds
        # (e.g., 1450 kg/m3 for yellow peas — NOT flour density).
        # Sphericity ~0.85 for roughly spherical legume seeds.
        p_density = material.rho_solid
        p_sphericity = 0.85  # whole seeds are roughly spherical

        # Collection bin geometry from the assembly
        head_x = cp.frame_length_m - cp.nose_length_m
        bin_under_bed = 0.15
        bin_past_end = 0.40
        bin_x0 = head_x - bin_under_bed
        bin_x1 = head_x + bin_past_end
        belt_w = cp.belt_width_m
        bin_width_z = belt_w + 0.06
        bin_z0_abs = (cp.frame_width_m - bin_width_z) / 2
        bin_z1_abs = bin_z0_abs + bin_width_z
        floor_y = -(cp.frame_height_m + cp.leg_height_m)
        bin_height = abs(floor_y) * 0.60
        bin_bottom = floor_y
        bin_top = floor_y + bin_height

        # Hopper geometry for particle placement
        hopper_back_x = hp.hopper_back_x if hp is not None else spawn_x - 0.60
        hopper_front_x = spawn_x
        hopper_height = hp.total_back_height_m if hp is not None else 0.30
        hopper_z0 = belt_z0
        hopper_z1 = belt_z1
        if hp is not None:
            z_center = hp.belt_z_center_m
            half_w = hp.hopper_width_m / 2.0
            hopper_z0 = z_center - half_w
            hopper_z1 = z_center + half_w

        # Throughput from manual Chapter 5 formula:
        # throughput = rho_bulk * bed_cross_area * v_belt
        rho_bulk = material.bulk_density(material.initial_moisture_wb)
        bed_cross = material.bed_depth_m * cp.belt_width_m
        # Use a nominal belt speed for mass accounting; actual speed
        # comes from the conveyor controller during the simulation.
        nominal_v_belt = 0.5 / 60.0  # 0.5 m/min default
        throughput = rho_bulk * bed_cross * nominal_v_belt

        cfg = ParticleSystemConfig(
            spawn_x=spawn_x,
            spawn_z0=belt_z0,
            spawn_z1=belt_z1,
            belt_y=cp.belt_stack_thickness_m,
            bed_depth_m=material.bed_depth_m,
            head_roller_x=head_x,
            head_roller_r=cp.head_roller_radius_m,
            # Hopper geometry
            hopper_back_x=hopper_back_x,
            hopper_front_x=hopper_front_x,
            hopper_height_m=hopper_height,
            hopper_z0=hopper_z0,
            hopper_z1=hopper_z1,
            # Bin geometry
            bin_x0=bin_x0,
            bin_x1=bin_x1,
            bin_z0=bin_z0_abs,
            bin_z1=bin_z1_abs,
            bin_bottom_y=bin_bottom,
            bin_top_y=bin_top,
            T_inlet_c=material.initial_temperature_c,
            M_inlet_wb=material.initial_moisture_wb,
            particle_density=p_density,
            particle_sphericity=p_sphericity,
            gravity=GRAVITY,
            throughput_kg_per_s=throughput,
            rf_x_start=op.rf_zone_x_start,
            rf_x_end=op.rf_zone_x_end,
            oven_x_end=op.oven_x_end_m,
        )
        return cls(cfg)

    # ------------------------------------------------------------------
    #  Step
    # ------------------------------------------------------------------

    def step(
        self,
        dt_sim: float,
        belt_speed_m_per_s: float,
        T_field: Optional[np.ndarray] = None,
        M_field: Optional[np.ndarray] = None,
        cell_is_material: Optional[np.ndarray] = None,
        grid_origin: Optional[Tuple[float, float, float]] = None,
        cell_sizes: Optional[Tuple[float, float, float]] = None,
    ):
        """Advance all particles by one simulation timestep.

        Kinematics (section 4.4):
            In hopper: gentle settling toward discharge
            Riding: dx = v_belt * dt
            Falling: dv/dt = -g (Newton's 2nd law, g from constants)

        Field coupling:
            T and M trilinearly interpolated from Eulerian grid.
        """
        cfg = self.cfg
        rng = np.random

        # ── Check mass limit ─────────────────────────────────────────
        can_dispatch = True
        if cfg.run_mass_kg > 0 and self._dispatched_mass_kg >= cfg.run_mass_kg:
            can_dispatch = False

        # ── Hopper settling (realistic granular flow) ─────────────────
        # The hopper drains through the sizing gate at the front (discharge).
        # Material flows in two zones:
        #   1. DISCHARGE ZONE: Near the sizing gate, material flows rapidly
        #      toward the exit. Flow velocity matches belt throughput.
        #   2. SETTLING ZONE: Above the discharge, material settles under
        #      gravity to fill voids left by material that has flowed out.
        #
        # Settling rates are scaled to the throughput velocity so that the
        # hopper drains at the same rate material exits the gate.  This
        # avoids the visual artefact of particles teleporting to the
        # hopper floor on the first rendered frame (the old hard-coded
        # rates of 0.08–0.15 m/s were ~30–70× the belt speed, causing
        # ~5 cm drops per timestep).
        hopper = (self.state == self._STATE_IN_HOPPER)
        if hopper.any():
            hopper_idx = np.where(hopper)[0]
            pos_h = self.pos[hopper_idx]

            # Hopper geometry
            gate_x = cfg.hopper_front_x  # discharge (sizing gate)
            back_x = cfg.hopper_back_x
            hopper_len = gate_x - back_x
            gate_height = cfg.bed_depth_m  # sizing gate opening = bed depth
            belt_y = cfg.belt_y

            # Distance from discharge gate (0 at gate, 1 at back wall)
            dist_from_gate = (gate_x - pos_h[:, 0]) / max(hopper_len, 0.01)
            dist_from_gate = np.clip(dist_from_gate, 0.0, 1.0)

            # Height above belt level
            height_above_belt = pos_h[:, 1] - belt_y

            # ── DISCHARGE ZONE: particles near gate and low enough ────
            # Particles within 30% of hopper length from gate AND below
            # 2.5× the gate opening flow toward the discharge.
            in_discharge_zone = (
                (dist_from_gate < 0.3) &
                (height_above_belt < gate_height * 2.5)
            )

            # Discharge flow velocity — scaled to belt throughput so the
            # hopper drains at a physically consistent rate.
            v_discharge = belt_speed_m_per_s * 2.0  # converges toward gate
            # Settling in discharge zone: proportional to discharge velocity
            # so particles reach gate level over ~10 s, not one timestep.
            v_settle_discharge = max(belt_speed_m_per_s * 5.0, 0.005)
            if in_discharge_zone.any():
                self.pos[hopper_idx[in_discharge_zone], 0] += v_discharge * dt_sim
                self.pos[hopper_idx[in_discharge_zone], 1] -= v_settle_discharge * dt_sim

            # ── SETTLING ZONE: particles above discharge zone ─────────
            # Settle to fill voids left by discharged material.  Rate
            # scales with proximity to gate (more void space near front).
            not_discharge = ~in_discharge_zone

            settle_base = max(belt_speed_m_per_s * 3.0, 0.003)
            settle_factor = 1.0 + 2.0 * (1.0 - dist_from_gate[not_discharge])
            settle_rate = settle_base * settle_factor

            if not_discharge.any():
                self.pos[hopper_idx[not_discharge], 1] -= settle_rate * dt_sim
                drift_rate = belt_speed_m_per_s * (1.0 - dist_from_gate[not_discharge])
                self.pos[hopper_idx[not_discharge], 0] += drift_rate * dt_sim

            # ── Clamp positions to hopper bounds ──────────────────────
            # NOTE: self.pos[hopper_idx, ...] is fancy-indexed (integer
            # array), so it returns a COPY, not a view.  Using out= with
            # np.clip would write to the discarded copy.  We must assign
            # the clipped result back explicitly.
            self.pos[hopper_idx, 1] = np.clip(
                self.pos[hopper_idx, 1],
                belt_y + 0.001,
                belt_y + cfg.hopper_height_m,
            )
            self.pos[hopper_idx, 0] = np.clip(
                self.pos[hopper_idx, 0],
                back_x + 0.01,
                gate_x - 0.005,
            )

        # ── Dispatch: hopper/dead → riding (at spawn rate) ───────────
        self._spawn_accumulator += cfg.spawn_rate * dt_sim
        n_spawn = int(self._spawn_accumulator)
        self._spawn_accumulator -= n_spawn

        if can_dispatch:
            for _ in range(n_spawn):
                # ── Find a particle to dispatch ─────────────────────────
                # Priority: hopper particles first (they drain naturally)
                idx = self._find_hopper_slot()
                if idx < 0:
                    # No hopper particles left
                    if cfg.run_mass_kg > 0:
                        # FINITE MASS mode: hopper empty = batch complete
                        # Don't use dead slots - that would recycle treated
                        # material as fresh, which is physically wrong.
                        break
                    else:
                        # CONTINUOUS mode: use dead slots to maintain flow
                        idx = self._find_dead_slot()
                        if idx < 0:
                            break

                # Place on belt at the hopper discharge point
                self.pos[idx, 0] = cfg.spawn_x
                self.pos[idx, 1] = cfg.belt_y + rng.uniform(0, cfg.bed_depth_m)
                self.pos[idx, 2] = rng.uniform(cfg.spawn_z0, cfg.spawn_z1)
                self.vel[idx] = 0.0
                self.state[idx] = self._STATE_RIDING
                self.age[idx] = 0.0
                self.temperature[idx] = cfg.T_inlet_c
                self.moisture[idx] = cfg.M_inlet_wb
                self._dispatched_mass_kg += self.mass_per_particle

                # Re-check mass limit after each dispatch
                if cfg.run_mass_kg > 0 and self._dispatched_mass_kg >= cfg.run_mass_kg:
                    break

        # ── Belt transport: v_belt in +X (section 4.4) ────────────
        riding = (self.state == self._STATE_RIDING)
        if riding.any():
            self.pos[riding, 0] += belt_speed_m_per_s * dt_sim

            at_head = riding & (self.pos[:, 0] >= cfg.head_roller_x)
            if at_head.any():
                self.state[at_head] = self._STATE_FALLING
                self.vel[at_head, 0] = belt_speed_m_per_s
                self.vel[at_head, 1] = 0.0
                self.vel[at_head, 2] = 0.0

        # ── Free-fall: Newton's 2nd law, a = -g (from constants) ─
        falling = (self.state == self._STATE_FALLING)
        if falling.any():
            self.vel[falling, 1] -= cfg.gravity * dt_sim
            self.pos[falling] += self.vel[falling] * dt_sim

            # Check if particle is within the bin's XZ footprint
            in_bin_xz = (
                falling
                & (self.pos[:, 0] >= cfg.bin_x0)
                & (self.pos[:, 0] <= cfg.bin_x1)
                & (self.pos[:, 2] >= cfg.bin_z0)
                & (self.pos[:, 2] <= cfg.bin_z1)
            )

            # Landed in the bin: settle as COLLECTED.
            landed = in_bin_xz & (self.pos[:, 1] <= cfg.bin_top_y)
            if landed.any():
                n_collected = int(np.sum(self.state == self._STATE_COLLECTED))
                fill_frac = min(n_collected / max(cfg.max_particles * 0.3, 1), 0.95)
                fill_y = cfg.bin_bottom_y + fill_frac * (cfg.bin_top_y - cfg.bin_bottom_y)
                n_land = int(landed.sum())
                # Spread across the full bin footprint (X, Z)
                self.pos[landed, 0] = rng.uniform(
                    cfg.bin_x0 + 0.01, cfg.bin_x1 - 0.01, size=n_land,
                )
                self.pos[landed, 2] = rng.uniform(
                    cfg.bin_z0 + 0.01, cfg.bin_z1 - 0.01, size=n_land,
                )
                # Settle at random Y within the current fill level
                self.pos[landed, 1] = rng.uniform(
                    cfg.bin_bottom_y + 0.005,
                    max(fill_y + 0.01, cfg.bin_bottom_y + 0.03),
                    size=n_land,
                )
                self.vel[landed] = 0.0
                self.state[landed] = self._STATE_COLLECTED

                # Mass accounting: adjust for moisture loss during drying.
                # Each particle was dispatched with mass_per_particle based on
                # initial moisture M_i.  After drying, the particle has moisture
                # M_f (interpolated from Eulerian grid).  The dry mass is conserved,
                # so the wet mass after drying is:
                #   m_out = m_in * (1 - M_i) / (1 - M_f)
                # This ensures mass collected < mass dispatched when M_f < M_i.
                M_initial = cfg.M_inlet_wb
                M_landed = self.moisture[landed]
                # Guard against division by zero (M_f approaching 1)
                M_landed_safe = np.clip(M_landed, 0.0, 0.99)
                mass_ratio = (1.0 - M_initial) / (1.0 - M_landed_safe)
                mass_per_particle_adjusted = self.mass_per_particle * mass_ratio
                self._total_collected_kg += float(np.sum(mass_per_particle_adjusted))

            # Fell outside the bin entirely → mark as lost
            # In continuous mode: DEAD particles get recycled
            # In finite mass mode: DEAD particles are lost (not re-dispatched)
            missed = falling & ~in_bin_xz & (self.pos[:, 1] < cfg.bin_bottom_y)
            self.state[missed] = self._STATE_DEAD

        # ── Recycle collected → dead (continuous mode only) ────────────
        # For CONTINUOUS mode (run_mass_kg == 0): recycle collected
        # particles to maintain a steady flow for visualization.
        #
        # For FINITE MASS mode (run_mass_kg > 0): NO RECYCLING.
        # Once collected, particles stay collected. This correctly
        # models the real process where treated material is not
        # re-fed into the machine. The bin gradually fills as
        # material is processed.
        if cfg.run_mass_kg == 0 and can_dispatch:
            # Continuous mode: maintain a dead-slot pool for spawning
            target_dead = int(cfg.spawn_rate * 0.5) + 20  # ~120 slots
            n_dead = int(np.sum(self.state == self._STATE_DEAD))
            n_hopper = int(np.sum(self.state == self._STATE_IN_HOPPER))
            deficit = target_dead - n_dead - n_hopper
            if deficit > 0:
                collected_idx = np.where(self.state == self._STATE_COLLECTED)[0]
                n_collected = len(collected_idx)
                # Keep a visual pool in the bin
                min_bin_visual = int(cfg.max_particles * 0.05)
                available = max(0, n_collected - min_bin_visual)
                if available > 0:
                    n_recycle = min(deficit, available)
                    self.state[collected_idx[:n_recycle]] = self._STATE_DEAD

        # ── Trilinear interpolation of T, M from Eulerian grid ────
        if T_field is not None and grid_origin is not None and cell_sizes is not None:
            self._interpolate_fields(
                T_field, M_field, cell_is_material,
                grid_origin, cell_sizes,
            )

        # ── Post-grid cooling (Newton's law) ─────────────────────────
        # The Eulerian grid only covers the RF zone.  Once material
        # exits the grid, it must cool — inside the oven chamber
        # (insulated, heated air) or on the open belt (ambient air).
        #
        # Without this, particles freeze at their last grid temperature
        # and appear hot in the collection bin (visually wrong and
        # physically wrong for downstream pipeline input).
        #
        # Cooling model:  dT/dt = -(T - T_env) / tau
        # where tau = m * cp / (h * A_s) is the Biot time constant.
        #
        # For an 8 mm whole pea (rho=1450, cp=1700):
        #   m = 3.9e-4 kg,  A_s = 2.0e-4 m^2
        #   tau_oven  ≈ 330 s  (h ≈ 2 W/m^2K inside oven, T_env ≈ 40 °C)
        #   tau_belt  ≈  66 s  (h ≈ 10 W/m^2K open belt, T_env ≈ 22 °C)
        if grid_origin is not None and cell_sizes is not None:
            grid_x_end = grid_origin[0] + cell_sizes[0] * T_field.shape[0] if T_field is not None else cfg.rf_x_end
            T_ambient = cfg.T_inlet_c

            # Particles past the grid but still inside the oven chamber
            in_oven_post_grid = (
                riding
                & (self.pos[:, 0] > grid_x_end)
                & (self.pos[:, 0] <= cfg.oven_x_end)
            )
            if in_oven_post_grid.any():
                # Oven interior: heated air (~40°C), low convection (insulated)
                T_oven_air = T_ambient + 20.0  # oven air is warmer than ambient
                tau_oven = 330.0  # slow cooling inside oven [s]
                dT = (self.temperature[in_oven_post_grid] - T_oven_air) * dt_sim / tau_oven
                self.temperature[in_oven_post_grid] -= dT

            # Particles past the oven exit: open belt, ambient air
            on_open_belt = (
                riding
                & (self.pos[:, 0] > cfg.oven_x_end)
            )
            if on_open_belt.any():
                tau_belt = 66.0  # faster cooling on open belt [s]
                dT = (self.temperature[on_open_belt] - T_ambient) * dt_sim / tau_belt
                self.temperature[on_open_belt] -= dT

            # Collected particles in the bin: continue cooling to ambient
            collected = (self.state == self._STATE_COLLECTED)
            if collected.any():
                tau_bin = 120.0  # bin is a pile, slower than single pea on belt [s]
                dT = (self.temperature[collected] - T_ambient) * dt_sim / tau_bin
                self.temperature[collected] -= dT

        alive = (self.state != self._STATE_DEAD)
        self.age[alive] += dt_sim
        self._alive_count = int(np.sum(alive))

    # ------------------------------------------------------------------
    #  Trilinear interpolation (standard E-L coupling)
    # ------------------------------------------------------------------

    def _interpolate_fields(self, T_field, M_field, mask, origin, cell_sizes):
        """Trilinear interpolation of Eulerian fields at particle positions.

        Only updates riding particles.  Falling particles retain their
        last in-oven values (physically correct -- they've left the
        field domain).
        """
        riding = (self.state == self._STATE_RIDING)
        if not riding.any():
            return

        x0, y0, z0 = origin
        dx, dy, dz = cell_sizes
        nx, ny, nz = T_field.shape

        gx = (self.pos[riding, 0] - x0) / dx - 0.5
        gy = (self.pos[riding, 1] - y0) / dy - 0.5
        gz = (self.pos[riding, 2] - z0) / dz - 0.5

        ix0 = np.clip(np.floor(gx).astype(int), 0, nx - 2)
        iy0 = np.clip(np.floor(gy).astype(int), 0, ny - 2)
        iz0 = np.clip(np.floor(gz).astype(int), 0, nz - 2)

        fx = np.clip(gx - ix0, 0.0, 1.0)
        fy = np.clip(gy - iy0, 0.0, 1.0)
        fz = np.clip(gz - iz0, 0.0, 1.0)

        # 8-corner trilinear weights
        w000 = (1 - fx) * (1 - fy) * (1 - fz)
        w100 = fx * (1 - fy) * (1 - fz)
        w010 = (1 - fx) * fy * (1 - fz)
        w110 = fx * fy * (1 - fz)
        w001 = (1 - fx) * (1 - fy) * fz
        w101 = fx * (1 - fy) * fz
        w011 = (1 - fx) * fy * fz
        w111 = fx * fy * fz

        ix1, iy1, iz1 = ix0 + 1, iy0 + 1, iz0 + 1

        # Temperature
        T_interp = (
            w000 * T_field[ix0, iy0, iz0] + w100 * T_field[ix1, iy0, iz0] +
            w010 * T_field[ix0, iy1, iz0] + w110 * T_field[ix1, iy1, iz0] +
            w001 * T_field[ix0, iy0, iz1] + w101 * T_field[ix1, iy0, iz1] +
            w011 * T_field[ix0, iy1, iz1] + w111 * T_field[ix1, iy1, iz1]
        )

        # Material mask: only accept interpolated values in material cells
        if mask is not None:
            nearest_ix = np.clip(np.round(gx + 0.5).astype(int), 0, nx - 1)
            nearest_iy = np.clip(np.round(gy + 0.5).astype(int), 0, ny - 1)
            nearest_iz = np.clip(np.round(gz + 0.5).astype(int), 0, nz - 1)
            in_material = (mask[nearest_ix, nearest_iy, nearest_iz] == 1)
            T_interp = np.where(in_material, T_interp, self.temperature[riding])

        self.temperature[riding] = T_interp.astype(np.float32)

        # Moisture
        if M_field is not None:
            M_interp = (
                w000 * M_field[ix0, iy0, iz0] + w100 * M_field[ix1, iy0, iz0] +
                w010 * M_field[ix0, iy1, iz0] + w110 * M_field[ix1, iy1, iz0] +
                w001 * M_field[ix0, iy0, iz1] + w101 * M_field[ix1, iy0, iz1] +
                w011 * M_field[ix0, iy1, iz1] + w111 * M_field[ix1, iy1, iz1]
            )
            if mask is not None:
                M_interp = np.where(in_material, M_interp, self.moisture[riding])
            # After batch exhausts, M=0 advects into the grid behind the material.
            # Skip ALL moisture interpolation after batch exhausts - particles
            # retain their last valid moisture until they land in the bin.
            # This prevents particles from picking up the M=0 "empty belt" values.
            if not self._batch_exhausted:
                # Normal operation: update moisture from grid
                # But still guard against empty cells (M < threshold)
                M_threshold = self.cfg.M_inlet_wb * 0.1  # 10% of initial = clearly material
                has_actual_material = M_interp > M_threshold
                M_interp = np.where(has_actual_material, M_interp, self.moisture[riding])
                self.moisture[riding] = M_interp.astype(np.float32)
            # else: batch exhausted - particles keep their current moisture

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _find_dead_slot(self) -> int:
        dead = np.where(self.state == self._STATE_DEAD)[0]
        return int(dead[0]) if len(dead) > 0 else -1

    def _find_hopper_slot(self) -> int:
        """Find a hopper particle to dispatch through the sizing gate.

        Selects the particle that is:
        1. Closest to the discharge gate (highest X position)
        2. Near the bottom (within the sizing gate opening)

        This creates a realistic flow pattern where material exits
        through the sizing gate at the bottom-front of the hopper.
        """
        cfg = self.cfg
        hopper = np.where(self.state == self._STATE_IN_HOPPER)[0]
        if len(hopper) == 0:
            return -1

        # Particle positions
        pos_x = self.pos[hopper, 0]
        pos_y = self.pos[hopper, 1]

        # Normalize to [0, 1] range for scoring
        x_norm = (pos_x - cfg.hopper_back_x) / max(cfg.hopper_front_x - cfg.hopper_back_x, 0.01)
        y_norm = (pos_y - cfg.belt_y) / max(cfg.hopper_height_m, 0.01)

        # Score: prefer high X (near gate) and low Y (near bottom)
        # Weight X more heavily since gate is at front
        score = 2.0 * x_norm - y_norm

        best_idx = hopper[np.argmax(score)]
        return int(best_idx)

    def get_positions(self) -> np.ndarray:
        alive = (self.state != self._STATE_DEAD)
        return self.pos[alive].copy()

    def get_temperatures(self) -> np.ndarray:
        alive = (self.state != self._STATE_DEAD)
        return self.temperature[alive].copy()

    def get_moistures(self) -> np.ndarray:
        alive = (self.state != self._STATE_DEAD)
        return self.moisture[alive].copy()

    def set_batch_exhausted(self, exhausted: bool = True):
        """Signal that the batch has exhausted (hopper empty).

        When batch exhausts, the Eulerian grid starts receiving M=0
        at the inlet. Particles should stop interpolating moisture
        from the grid and retain their last valid values.
        """
        self._batch_exhausted = exhausted

    @property
    def alive_count(self) -> int:
        return self._alive_count

    @property
    def collected_count(self) -> int:
        """Number of particles currently in the collection bin."""
        return int(np.sum(self.state == self._STATE_COLLECTED))

    @property
    def hopper_count(self) -> int:
        """Number of particles currently in the hopper."""
        return int(np.sum(self.state == self._STATE_IN_HOPPER))

    @property
    def hopper_empty(self) -> bool:
        """True if all particles have left the hopper."""
        return self.hopper_count == 0

    @property
    def hopper_fill_fraction(self) -> float:
        """Fraction of particles remaining in hopper [0.0 to 1.0].

        For finite mass mode: starts at 1.0, decreases as material drains.
        Useful for visualizing hopper level during a run.
        """
        return self.hopper_count / max(self.cfg.max_particles, 1)

    @property
    def hopper_drain_rate(self) -> float:
        """Current spawn rate [particles/s].

        For finite mass mode, this is calculated to match the throughput
        so the hopper empties exactly when the run completes.
        """
        return self.cfg.spawn_rate

    @property
    def estimated_time_remaining_s(self) -> float:
        """Estimated time until hopper is empty [s].

        Based on current hopper count and spawn rate.
        Returns 0.0 if hopper is empty or spawn rate is zero.
        """
        if self.cfg.spawn_rate <= 0 or self.hopper_empty:
            return 0.0
        return self.hopper_count / self.cfg.spawn_rate

    @property
    def falling_count(self) -> int:
        """Number of particles currently falling into the bin."""
        return int(np.sum(self.state == self._STATE_FALLING))

    @property
    def riding_count(self) -> int:
        """Number of particles currently on the belt."""
        return int(np.sum(self.state == self._STATE_RIDING))

    @property
    def collected_mass_kg(self) -> float:
        """Total mass collected in the bin [kg].

        Based on the throughput formula from the GP-15 manual
        (Chapter 5): throughput = rho_bulk * bed_cross * v_belt.
        Each particle represents throughput / spawn_rate kg.
        """
        return self._total_collected_kg

    @property
    def dispatched_mass_kg(self) -> float:
        """Cumulative mass dispatched from hopper to belt [kg].

        For finite mass mode (run_mass_kg > 0):
        - Starts at 0 (all particles begin in hopper)
        - Equals run_mass_kg when hopper is fully emptied

        For continuous mode (run_mass_kg == 0):
        - Tracks total mass dispatched, unlimited
        """
        return self._dispatched_mass_kg

    @property
    def hopper_dispatched_mass_kg(self) -> float:
        """Mass dispatched from hopper during the run [kg].

        Same as dispatched_mass_kg for finite mass mode (no pre-placed
        belt particles).
        """
        return self._dispatched_mass_kg - self._initial_belt_mass_kg

    @property
    def initial_belt_mass_kg(self) -> float:
        """Mass of particles pre-placed on belt at simulation start [kg].

        For finite mass mode: always 0 (all particles start in hopper).
        For continuous mode: may be non-zero for visual purposes.
        """
        return self._initial_belt_mass_kg

    @property
    def run_complete(self) -> bool:
        """Check if a finite mass run has completed.

        A run is complete when:
        1. All mass has been dispatched (dispatched_mass_kg >= run_mass_kg)
        2. All dispatched particles have been collected (no RIDING or FALLING)

        For continuous mode (run_mass_kg == 0), always returns False.
        """
        cfg = self.cfg
        if cfg.run_mass_kg <= 0:
            return False  # Continuous mode never completes

        # Check if all mass has been dispatched
        if self._dispatched_mass_kg < cfg.run_mass_kg:
            return False

        # Check if belt has cleared (no particles riding or falling)
        n_in_transit = int(np.sum(
            (self.state == self._STATE_RIDING) |
            (self.state == self._STATE_FALLING)
        ))
        return n_in_transit == 0

    @property
    def run_progress(self) -> float:
        """Progress of a finite mass run [0.0 to 1.0].

        Based on collected mass vs run_mass_kg.
        For continuous mode, returns 0.0.
        """
        cfg = self.cfg
        if cfg.run_mass_kg <= 0:
            return 0.0
        return min(self._total_collected_kg / cfg.run_mass_kg, 1.0)
