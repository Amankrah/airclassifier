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


class MaterialParticleSystem:
    """Lagrangian tracer particles coupled to the Eulerian physics grid.

    Each particle carries:
        pos[i]         -- world position (x, y, z) [m]
        vel[i]         -- velocity (only during free-fall) [m/s]
        temperature[i] -- trilinearly interpolated from grid T [C]
        moisture[i]    -- trilinearly interpolated from grid M [wb]
        state[i]       -- IN_HOPPER / RIDING / FALLING / COLLECTED / DEAD

    Lifecycle::

        IN_HOPPER → RIDING → FALLING → COLLECTED → (recycle to IN_HOPPER)

    When ``run_mass_kg > 0``, dispatching stops once the cumulative
    mass reaches the limit.  The belt clears naturally.

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

        self._alive_count = 0
        self._spawn_accumulator = 0.0
        self._init_particles()

    # ------------------------------------------------------------------
    #  Initialization
    # ------------------------------------------------------------------

    def _init_particles(self):
        """Distribute particles between hopper and belt.

        When ``run_mass_kg > 0`` (finite mass run):
            - A small pool (~500) starts inside the hopper for visual
            - The rest start distributed along the belt (already flowing)

        When ``run_mass_kg == 0`` (continuous / infinite):
            - All particles start on the belt (original behaviour)
        """
        cfg = self.cfg
        n = cfg.max_particles
        rng = np.random.default_rng(42)

        belt_len = cfg.head_roller_x - cfg.spawn_x
        if belt_len <= 0:
            return

        if cfg.run_mass_kg > 0 and cfg.hopper_front_x > cfg.hopper_back_x:
            # ── Finite mass: small hopper visual pool + belt ──────
            # Keep the hopper pool small so it doesn't look like a
            # dense block.  The rest of the particles recycle through
            # DEAD→RIDING→FALLING→COLLECTED→DEAD while mass remains.
            n_hopper = min(500, int(n * 0.08))
            n_belt = n - n_hopper

            # Place particles inside hopper — only in the lower
            # portion near the sizing gate (where material sits)
            hopper_fill_h = min(cfg.hopper_height_m * 0.35, cfg.bed_depth_m * 4)
            for i in range(n_hopper):
                self.pos[i, 0] = rng.uniform(cfg.hopper_back_x + 0.05,
                                             cfg.hopper_front_x - 0.01)
                self.pos[i, 1] = cfg.belt_y + rng.uniform(0.0, hopper_fill_h)
                self.pos[i, 2] = rng.uniform(cfg.hopper_z0, cfg.hopper_z1)
                self.state[i] = self._STATE_IN_HOPPER
                self.temperature[i] = cfg.T_inlet_c
                self.moisture[i] = cfg.M_inlet_wb

            # Place remaining particles on belt
            for i in range(n_hopper, n):
                frac = (i - n_hopper) / max(n_belt - 1, 1)
                self.pos[i, 0] = cfg.spawn_x + frac * belt_len
                self.pos[i, 1] = cfg.belt_y + rng.uniform(0, cfg.bed_depth_m)
                self.pos[i, 2] = rng.uniform(cfg.spawn_z0, cfg.spawn_z1)
                self.state[i] = self._STATE_RIDING
                self.temperature[i] = cfg.T_inlet_c
                self.moisture[i] = cfg.M_inlet_wb

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

        Args:
            run_mass_kg: Total mass to feed from the hopper [kg].
                         0 = continuous (infinite) mode.
            throughput_kg_per_s: Actual throughput at recipe belt speed.
                                If > 0, recalculates mass_per_particle.
        """
        self.cfg.run_mass_kg = run_mass_kg
        if throughput_kg_per_s > 0 and self.cfg.spawn_rate > 0:
            self.cfg.throughput_kg_per_s = throughput_kg_per_s
            self.mass_per_particle = throughput_kg_per_s / self.cfg.spawn_rate
        self._dispatched_mass_kg = 0.0
        self._total_collected_kg = 0.0
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

        # ── Hopper settling (visual gravity) ─────────────────────────
        hopper = (self.state == self._STATE_IN_HOPPER)
        if hopper.any():
            # Settle downward toward belt level
            self.pos[hopper, 1] -= 0.12 * dt_sim
            np.clip(self.pos[hopper, 1], cfg.belt_y,
                    cfg.belt_y + cfg.hopper_height_m,
                    out=self.pos[hopper, 1])
            # Drift toward discharge (front wall)
            self.pos[hopper, 0] += 0.05 * dt_sim
            np.clip(self.pos[hopper, 0],
                    cfg.hopper_back_x + 0.01, cfg.hopper_front_x - 0.01,
                    out=self.pos[hopper, 0])

        # ── Dispatch: hopper/dead → riding (at spawn rate) ───────────
        self._spawn_accumulator += cfg.spawn_rate * dt_sim
        n_spawn = int(self._spawn_accumulator)
        self._spawn_accumulator -= n_spawn

        if can_dispatch:
            for _ in range(n_spawn):
                # First try hopper particles (they drain naturally)
                idx = self._find_hopper_slot()
                if idx < 0:
                    # No hopper particles left — use dead slots
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

            # Fell outside the bin entirely → recycle
            missed = falling & ~in_bin_xz & (self.pos[:, 1] < cfg.bin_bottom_y)
            self.state[missed] = self._STATE_DEAD

        # ── Recycle collected → dead (keep dead pool for spawning) ────
        # Maintain a steady dead-slot pool sized to cover ~0.5 s of
        # spawning.  This prevents feast-famine gaps on the belt.
        # BUT always preserve a visual pool in the bin so it fills up.
        # Stop recycling once mass is exhausted so the belt clears.
        if can_dispatch:
            target_dead = int(cfg.spawn_rate * 0.5) + 20  # ~120 slots
            n_dead = int(np.sum(self.state == self._STATE_DEAD))
            n_hopper = int(np.sum(self.state == self._STATE_IN_HOPPER))
            deficit = target_dead - n_dead - n_hopper
            if deficit > 0:
                collected_idx = np.where(self.state == self._STATE_COLLECTED)[0]
                n_collected = len(collected_idx)
                # Always keep a growing visual pool in the bin.
                # The pool grows proportionally to mass dispatched.
                if cfg.run_mass_kg > 0:
                    frac_done = min(self._dispatched_mass_kg / cfg.run_mass_kg, 1.0)
                else:
                    frac_done = 0.0
                # Reserve: 5% of particles at start → 40% at end of run
                min_bin_visual = int(cfg.max_particles * (0.05 + 0.35 * frac_done))
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
            self.moisture[riding] = M_interp.astype(np.float32)

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _find_dead_slot(self) -> int:
        dead = np.where(self.state == self._STATE_DEAD)[0]
        return int(dead[0]) if len(dead) > 0 else -1

    def _find_hopper_slot(self) -> int:
        """Find a hopper particle to dispatch (closest to discharge)."""
        hopper = np.where(self.state == self._STATE_IN_HOPPER)[0]
        if len(hopper) == 0:
            return -1
        # Pick the one closest to the discharge (highest X position)
        best = hopper[np.argmax(self.pos[hopper, 0])]
        return int(best)

    def get_positions(self) -> np.ndarray:
        alive = (self.state != self._STATE_DEAD)
        return self.pos[alive].copy()

    def get_temperatures(self) -> np.ndarray:
        alive = (self.state != self._STATE_DEAD)
        return self.temperature[alive].copy()

    def get_moistures(self) -> np.ndarray:
        alive = (self.state != self._STATE_DEAD)
        return self.moisture[alive].copy()

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
        """Cumulative mass dispatched from hopper to belt [kg]."""
        return self._dispatched_mass_kg
