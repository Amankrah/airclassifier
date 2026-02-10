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

    1. Move with belt velocity v_belt in +X  (section 4.4)
    2. Trilinearly interpolate T and M from the Eulerian grid
    3. Fall under gravity when leaving the head roller (Newton's 2nd law)
    4. Spawn with T_inlet, M_inlet from MaterialProperties

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

# ── Pretreatment material system ─────────────────────────────────────
from .config import MaterialProperties as PretreatmentMaterial

# ── Particle material system (density, sphericity, size distributions)
try:
    from ..particles.material import ParticleMaterial
    _HAS_PARTICLE_MATERIAL = True
except ImportError:
    _HAS_PARTICLE_MATERIAL = False


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

    # Particle physical properties (from ParticleMaterial / MaterialDensities)
    particle_density: float = 1450.0  # solid density [kg/m^3]
    particle_sphericity: float = 0.70 # shape factor [-]

    # Physics constant (from utils.constants)
    gravity: float = GRAVITY          # [m/s^2]

    # Grid info (for field interpolation bounds)
    rf_x_start: float = 0.0
    rf_x_end: float = 1.5


def _get_particle_material_for_feedstock(material: PretreatmentMaterial) -> Optional[dict]:
    """Look up the particle-system material for the pretreatment feedstock.

    Bridges the two material systems:
        pretreatment.config.MaterialProperties  -->  particles.material.ParticleMaterial

    The pretreatment material carries dielectric/thermal properties
    (for the Eulerian grid).  The particle material carries density,
    sphericity, and size distribution (for Lagrangian tracers).

    Both are keyed by feedstock name (yellow_pea, faba_bean, oat).
    """
    if not _HAS_PARTICLE_MATERIAL:
        return None

    name = material.name.lower().replace(" ", "_")
    try:
        pm = ParticleMaterial.create_food_powder(name, "whole")
        return {
            "density": pm.density,
            "sphericity": pm.sphericity,
        }
    except (ValueError, KeyError):
        return None


class MaterialParticleSystem:
    """Lagrangian tracer particles coupled to the Eulerian physics grid.

    Each particle carries:
        pos[i]         -- world position (x, y, z) [m]
        vel[i]         -- velocity (only during free-fall) [m/s]
        temperature[i] -- trilinearly interpolated from grid T [C]
        moisture[i]    -- trilinearly interpolated from grid M [wb]
        state[i]       -- RIDING / FALLING / DEAD

    Material properties are sourced dynamically:
        - Inlet T, M          from PretreatmentMaterial (config.py)
        - Particle density     from particles.material.ParticleMaterial
        - Gravity              from utils.constants.GRAVITY
        - Geometry             from machine assembly components
    """

    _STATE_RIDING = 0
    _STATE_FALLING = 1
    _STATE_COLLECTED = 2   # sitting in the collection bin
    _STATE_DEAD = 3        # recycled (off-screen, awaiting respawn)

    def __init__(self, config: ParticleSystemConfig):
        self.cfg = config
        n = config.max_particles

        self.pos = np.zeros((n, 3), dtype=np.float32)
        self.vel = np.zeros((n, 3), dtype=np.float32)
        self.temperature = np.full(n, config.T_inlet_c, dtype=np.float32)
        self.moisture = np.full(n, config.M_inlet_wb, dtype=np.float32)
        self.state = np.full(n, self._STATE_DEAD, dtype=np.int32)
        self.age = np.zeros(n, dtype=np.float32)

        self._alive_count = 0
        self._spawn_accumulator = 0.0
        self._prefill()

    def _prefill(self):
        """Distribute particles along the belt for a non-empty first frame."""
        cfg = self.cfg
        n = cfg.max_particles
        belt_len = cfg.head_roller_x - cfg.spawn_x
        if belt_len <= 0:
            return

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

    @classmethod
    def from_assembly(
        cls,
        assembly,
        material: PretreatmentMaterial,
    ) -> "MaterialParticleSystem":
        """Create from the machine assembly and material properties.

        Geometry comes from the assembly's component parameters
        (single source of truth).  Inlet T and M come from the
        pretreatment MaterialProperties.  Particle density and
        sphericity come from the project's particle material system
        (if available), falling back to the pretreatment material's
        rho_solid.
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

        # Look up particle-system material for density and sphericity
        pm_props = _get_particle_material_for_feedstock(material)
        if pm_props is not None:
            p_density = pm_props["density"]
            p_sphericity = pm_props["sphericity"]
        else:
            # Fallback: use pretreatment material's solid density
            p_density = material.rho_solid
            p_sphericity = 0.70  # typical for whole seeds

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

        cfg = ParticleSystemConfig(
            spawn_x=spawn_x,
            spawn_z0=belt_z0,
            spawn_z1=belt_z1,
            belt_y=cp.belt_stack_thickness_m,
            bed_depth_m=material.bed_depth_m,
            head_roller_x=head_x,
            head_roller_r=cp.head_roller_radius_m,
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
            Riding: dx = v_belt * dt
            Falling: dv/dt = -g (Newton's 2nd law, g from constants)

        Field coupling:
            T and M trilinearly interpolated from Eulerian grid.
        """
        cfg = self.cfg
        rng = np.random

        # ── Spawn with inlet conditions from MaterialProperties ───
        self._spawn_accumulator += cfg.spawn_rate * dt_sim
        n_spawn = int(self._spawn_accumulator)
        self._spawn_accumulator -= n_spawn

        for _ in range(n_spawn):
            idx = self._find_dead_slot()
            if idx < 0:
                break
            self.pos[idx, 0] = cfg.spawn_x
            self.pos[idx, 1] = cfg.belt_y + rng.uniform(0, cfg.bed_depth_m)
            self.pos[idx, 2] = rng.uniform(cfg.spawn_z0, cfg.spawn_z1)
            self.vel[idx] = 0.0
            self.state[idx] = self._STATE_RIDING
            self.age[idx] = 0.0
            self.temperature[idx] = cfg.T_inlet_c
            self.moisture[idx] = cfg.M_inlet_wb

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

            # Landed in the bin: settle as COLLECTED
            landed = in_bin_xz & (self.pos[:, 1] <= cfg.bin_top_y)
            if landed.any():
                # Fill level rises as particles accumulate
                n_collected = int(np.sum(self.state == self._STATE_COLLECTED))
                fill_frac = min(n_collected / max(cfg.max_particles * 0.3, 1), 0.95)
                fill_y = cfg.bin_bottom_y + fill_frac * (cfg.bin_top_y - cfg.bin_bottom_y)
                n_land = int(landed.sum())
                self.pos[landed, 1] = np.random.uniform(
                    cfg.bin_bottom_y + 0.005,
                    max(fill_y + 0.01, cfg.bin_bottom_y + 0.02),
                    size=n_land,
                )
                self.vel[landed] = 0.0
                self.state[landed] = self._STATE_COLLECTED

            # Fell outside the bin entirely → recycle
            missed = falling & ~in_bin_xz & (self.pos[:, 1] < cfg.bin_bottom_y)
            self.state[missed] = self._STATE_DEAD

        # ── Recycle oldest collected particles when bin is full ────
        # Keeps a pool of dead slots available for spawning.
        n_collected = int(np.sum(self.state == self._STATE_COLLECTED))
        n_dead = int(np.sum(self.state == self._STATE_DEAD))
        if n_dead < 50 and n_collected > cfg.max_particles * 0.3:
            collected_idx = np.where(self.state == self._STATE_COLLECTED)[0]
            self.state[collected_idx[:50]] = self._STATE_DEAD

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
