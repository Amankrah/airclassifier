"""
CFD-DEM Coupling for cyclone air classifier simulation.

Provides two-way coupling between Eulerian CFD flow field solver
and Lagrangian particle tracking (DEM).

Coupling approaches:
- One-way: CFD affects particles, particles don't affect flow
- Two-way: Particles exert momentum source on fluid (for dense flows)
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Callable
from enum import Enum
import numpy as np
import warp as wp

from ..fluid.solvers.navier_stokes import (
    NavierStokesSolver,
    GridParams,
    FluidProperties,
    SolverParams,
)
from ..fluid.turbulence.models import (
    KEpsilonModel,
    KEpsilonParams,
    SmagorinskyModel,
    SmagorinskyParams,
)
from ..fluid.boundary_conditions import (
    BoundaryConditionManager,
    InletCondition,
    OutletCondition,
    WallCondition,
)
from ..particles.material import ParticleMaterial


class CouplingMode(Enum):
    """CFD-DEM coupling mode."""
    ONE_WAY = "one_way"      # CFD -> particles only
    TWO_WAY = "two_way"      # CFD <-> particles (momentum exchange)


class TurbulenceModelType(Enum):
    """Available turbulence models."""
    NONE = "none"
    K_EPSILON = "k_epsilon"
    SMAGORINSKY = "smagorinsky"


@dataclass
class CFDConfig:
    """Configuration for CFD solver."""

    # Grid parameters
    domain_size: Tuple[float, float, float] = (0.4, 1.5, 0.4)
    resolution: Tuple[int, int, int] = (64, 128, 64)

    # Fluid properties
    density: float = 1.2              # kg/m³
    kinematic_viscosity: float = 1.5e-5  # m²/s

    # Solver settings
    dt: float = 1.0e-4
    max_pressure_iterations: int = 100
    pressure_tolerance: float = 1.0e-6

    # Turbulence
    turbulence_model: TurbulenceModelType = TurbulenceModelType.K_EPSILON
    turbulence_intensity: float = 0.05
    turbulence_length_scale: float = 0.01

    # Coupling
    coupling_mode: CouplingMode = CouplingMode.ONE_WAY
    cfd_substeps: int = 10  # CFD steps per particle step


@dataclass
class DEMConfig:
    """Configuration for DEM (particle) solver."""

    dt: float = 1.0e-5
    num_particles: int = 5000
    injection_duration: float = 0.1

    # Wall collision
    wall_restitution: float = 0.7
    wall_friction: float = 0.3


@dataclass
class CycloneCFDParams:
    """Cyclone geometry parameters for CFD setup."""

    cylinder_diameter: float = 0.3
    cylinder_height: float = 0.45
    cone_height: float = 0.75
    cone_tip_diameter: float = 0.1125
    inlet_width: float = 0.075
    inlet_height: float = 0.15
    inlet_velocity: float = 15.0
    vortex_finder_diameter: float = 0.15
    vortex_finder_length: float = 0.15


# Warp kernels for CFD-DEM coupling

@wp.kernel
def interpolate_velocity_to_particles(
    positions: wp.array(dtype=wp.vec3),
    is_active: wp.array(dtype=wp.int32),
    vel_x: wp.array3d(dtype=float),
    vel_y: wp.array3d(dtype=float),
    vel_z: wp.array3d(dtype=float),
    fluid_velocities: wp.array(dtype=wp.vec3),
    domain_origin: wp.vec3,
    dx: float,
    dy: float,
    dz: float,
    nx: int,
    ny: int,
    nz: int,
):
    """
    Interpolate fluid velocity from grid to particle positions.

    Uses trilinear interpolation.
    """
    tid = wp.tid()

    if is_active[tid] != 1:
        fluid_velocities[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    pos = positions[tid]

    # Convert to grid coordinates
    gx = (pos[0] - domain_origin[0]) / dx
    gy = (pos[1] - domain_origin[1]) / dy
    gz = (pos[2] - domain_origin[2]) / dz

    # Clamp to valid range
    gx = wp.clamp(gx, 0.5, float(nx) - 1.5)
    gy = wp.clamp(gy, 0.5, float(ny) - 1.5)
    gz = wp.clamp(gz, 0.5, float(nz) - 1.5)

    # Integer indices
    i0 = int(wp.floor(gx))
    j0 = int(wp.floor(gy))
    k0 = int(wp.floor(gz))

    i1 = wp.min(i0 + 1, nx - 1)
    j1 = wp.min(j0 + 1, ny - 1)
    k1 = wp.min(k0 + 1, nz - 1)

    # Interpolation weights
    sx = gx - float(i0)
    sy = gy - float(j0)
    sz = gz - float(k0)

    # Trilinear interpolation for each velocity component
    vx = (
        vel_x[i0, j0, k0] * (1.0 - sx) * (1.0 - sy) * (1.0 - sz) +
        vel_x[i1, j0, k0] * sx * (1.0 - sy) * (1.0 - sz) +
        vel_x[i0, j1, k0] * (1.0 - sx) * sy * (1.0 - sz) +
        vel_x[i0, j0, k1] * (1.0 - sx) * (1.0 - sy) * sz +
        vel_x[i1, j1, k0] * sx * sy * (1.0 - sz) +
        vel_x[i1, j0, k1] * sx * (1.0 - sy) * sz +
        vel_x[i0, j1, k1] * (1.0 - sx) * sy * sz +
        vel_x[i1, j1, k1] * sx * sy * sz
    )

    vy = (
        vel_y[i0, j0, k0] * (1.0 - sx) * (1.0 - sy) * (1.0 - sz) +
        vel_y[i1, j0, k0] * sx * (1.0 - sy) * (1.0 - sz) +
        vel_y[i0, j1, k0] * (1.0 - sx) * sy * (1.0 - sz) +
        vel_y[i0, j0, k1] * (1.0 - sx) * (1.0 - sy) * sz +
        vel_y[i1, j1, k0] * sx * sy * (1.0 - sz) +
        vel_y[i1, j0, k1] * sx * (1.0 - sy) * sz +
        vel_y[i0, j1, k1] * (1.0 - sx) * sy * sz +
        vel_y[i1, j1, k1] * sx * sy * sz
    )

    vz = (
        vel_z[i0, j0, k0] * (1.0 - sx) * (1.0 - sy) * (1.0 - sz) +
        vel_z[i1, j0, k0] * sx * (1.0 - sy) * (1.0 - sz) +
        vel_z[i0, j1, k0] * (1.0 - sx) * sy * (1.0 - sz) +
        vel_z[i0, j0, k1] * (1.0 - sx) * (1.0 - sy) * sz +
        vel_z[i1, j1, k0] * sx * sy * (1.0 - sz) +
        vel_z[i1, j0, k1] * sx * (1.0 - sy) * sz +
        vel_z[i0, j1, k1] * (1.0 - sx) * sy * sz +
        vel_z[i1, j1, k1] * sx * sy * sz
    )

    fluid_velocities[tid] = wp.vec3(vx, vy, vz)


@wp.kernel
def compute_particle_momentum_source(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    diameters: wp.array(dtype=float),
    is_active: wp.array(dtype=wp.int32),
    fluid_velocities: wp.array(dtype=wp.vec3),
    momentum_x: wp.array3d(dtype=float),
    momentum_y: wp.array3d(dtype=float),
    momentum_z: wp.array3d(dtype=float),
    particle_count: wp.array3d(dtype=float),
    domain_origin: wp.vec3,
    dx: float,
    dy: float,
    dz: float,
    nx: int,
    ny: int,
    nz: int,
    fluid_density: float,
    fluid_viscosity: float,
    cell_volume: float,
):
    """
    Compute momentum source from particles to fluid (two-way coupling).

    Distributes drag force reaction to nearby grid cells.
    """
    tid = wp.tid()

    if is_active[tid] != 1:
        return

    pos = positions[tid]
    vel_p = velocities[tid]
    d = diameters[tid]
    vel_f = fluid_velocities[tid]

    # Relative velocity
    vel_rel = vel_f - vel_p
    vel_mag = wp.length(vel_rel)

    if vel_mag < 1.0e-10:
        return

    # Particle Reynolds number
    Re_p = fluid_density * vel_mag * d / fluid_viscosity

    # Schiller-Naumann drag coefficient
    if Re_p < 1000.0:
        C_D = (24.0 / Re_p) * (1.0 + 0.15 * wp.pow(Re_p, 0.687))
    else:
        C_D = 0.44

    # Drag force on particle
    area = 0.25 * 3.14159265359 * d * d
    F_drag_mag = 0.5 * C_D * fluid_density * area * vel_mag * vel_mag

    # Force vector (reaction on fluid is opposite)
    F_on_fluid = vel_rel * (-F_drag_mag / vel_mag)

    # Grid coordinates
    gx = (pos[0] - domain_origin[0]) / dx
    gy = (pos[1] - domain_origin[1]) / dy
    gz = (pos[2] - domain_origin[2]) / dz

    i = int(wp.floor(gx))
    j = int(wp.floor(gy))
    k = int(wp.floor(gz))

    if i >= 0 and i < nx and j >= 0 and j < ny and k >= 0 and k < nz:
        # Add momentum source (force per unit volume)
        wp.atomic_add(momentum_x, i, j, k, F_on_fluid[0] / cell_volume)
        wp.atomic_add(momentum_y, i, j, k, F_on_fluid[1] / cell_volume)
        wp.atomic_add(momentum_z, i, j, k, F_on_fluid[2] / cell_volume)
        wp.atomic_add(particle_count, i, j, k, 1.0)


@wp.kernel
def apply_momentum_source_to_velocity(
    vel_x: wp.array3d(dtype=float),
    vel_y: wp.array3d(dtype=float),
    vel_z: wp.array3d(dtype=float),
    momentum_x: wp.array3d(dtype=float),
    momentum_y: wp.array3d(dtype=float),
    momentum_z: wp.array3d(dtype=float),
    dt: float,
    fluid_density: float,
    nx: int,
    ny: int,
    nz: int,
):
    """Apply particle momentum source to fluid velocity."""
    i, j, k = wp.tid()

    if i >= nx or j >= ny or k >= nz:
        return

    # dv/dt = F/rho -> dv = (F/rho) * dt
    vel_x[i, j, k] = vel_x[i, j, k] + momentum_x[i, j, k] * dt / fluid_density
    vel_y[i, j, k] = vel_y[i, j, k] + momentum_y[i, j, k] * dt / fluid_density
    vel_z[i, j, k] = vel_z[i, j, k] + momentum_z[i, j, k] * dt / fluid_density


class CFDDEMCoupler:
    """
    Coupled CFD-DEM solver for cyclone air classifier.

    Manages the interaction between:
    - Eulerian CFD flow field (NavierStokesSolver)
    - Lagrangian particle tracking (DEM)
    - Turbulence modeling
    - Boundary conditions
    """

    def __init__(
        self,
        cyclone_params: CycloneCFDParams,
        cfd_config: CFDConfig,
        dem_config: DEMConfig,
        material: ParticleMaterial,
        device: str = "cuda"
    ):
        """
        Initialize coupled CFD-DEM solver.

        Args:
            cyclone_params: Cyclone geometry parameters
            cfd_config: CFD solver configuration
            dem_config: DEM (particle) configuration
            material: Particle material properties
            device: Warp device
        """
        self.cyclone = cyclone_params
        self.cfd_config = cfd_config
        self.dem_config = dem_config
        self.material = material
        self.device = device

        # Initialize CFD solver
        self._init_cfd_solver()

        # Initialize turbulence model
        self._init_turbulence()

        # Initialize boundary conditions
        self._init_boundary_conditions()

        # Initialize particle arrays
        self._init_particles()

        # Coupling arrays
        self._init_coupling_arrays()

        # State
        self._time = 0.0
        self._cfd_step = 0
        self._dem_step = 0

    def _init_cfd_solver(self):
        """Initialize the Navier-Stokes CFD solver."""
        grid_params = GridParams(
            domain_size=self.cfd_config.domain_size,
            resolution=self.cfd_config.resolution,
        )

        fluid_props = FluidProperties(
            density=self.cfd_config.density,
            kinematic_viscosity=self.cfd_config.kinematic_viscosity,
        )

        solver_params = SolverParams(
            dt=self.cfd_config.dt,
            max_iterations=self.cfd_config.max_pressure_iterations,
            tolerance=self.cfd_config.pressure_tolerance,
        )

        self.cfd_solver = NavierStokesSolver(
            grid_params=grid_params,
            fluid_props=fluid_props,
            solver_params=solver_params,
            device=self.device
        )

        # Initialize flow field
        self.cfd_solver.initialize_cyclone_flow(
            inlet_velocity=self.cyclone.inlet_velocity,
            cylinder_radius=self.cyclone.cylinder_diameter / 2,
            vortex_finder_radius=self.cyclone.vortex_finder_diameter / 2,
        )

        # Store grid info
        self.nx, self.ny, self.nz = self.cfd_config.resolution
        self.dx = self.cfd_config.domain_size[0] / self.nx
        self.dy = self.cfd_config.domain_size[1] / self.ny
        self.dz = self.cfd_config.domain_size[2] / self.nz

        # Domain origin (centered in X-Z, bottom in Y)
        self.domain_origin = wp.vec3(0.0, 0.0, 0.0)

    def _init_turbulence(self):
        """Initialize turbulence model."""
        self.turbulence_model = None

        if self.cfd_config.turbulence_model == TurbulenceModelType.K_EPSILON:
            params = KEpsilonParams()
            self.turbulence_model = KEpsilonModel(
                params=params,
                grid_shape=(self.nx, self.ny, self.nz),
                grid_spacing=(self.dx, self.dy, self.dz),
                molecular_viscosity=self.cfd_config.kinematic_viscosity,
                device=self.device
            )
            # Initialize turbulence fields
            k_init = 1.5 * (self.cfd_config.turbulence_intensity *
                          self.cyclone.inlet_velocity)**2
            eps_init = 0.09**(3/4) * k_init**(3/2) / self.cfd_config.turbulence_length_scale
            self.turbulence_model.initialize(k_init=k_init, epsilon_init=eps_init)

        elif self.cfd_config.turbulence_model == TurbulenceModelType.SMAGORINSKY:
            params = SmagorinskyParams()
            self.turbulence_model = SmagorinskyModel(
                params=params,
                grid_shape=(self.nx, self.ny, self.nz),
                grid_spacing=(self.dx, self.dy, self.dz),
                device=self.device
            )

    def _init_boundary_conditions(self):
        """Initialize boundary condition manager."""
        self.bc_manager = BoundaryConditionManager(
            grid_shape=(self.nx, self.ny, self.nz),
            grid_spacing=(self.dx, self.dy, self.dz),
            device=self.device
        )

        # Set cyclone boundaries
        self.bc_manager.set_cyclone_boundaries(
            cylinder_radius=self.cyclone.cylinder_diameter / 2,
            cone_height=self.cyclone.cone_height,
            cone_bottom_radius=self.cyclone.cone_tip_diameter / 2,
            cylinder_height=self.cyclone.cylinder_height,
            inlet_position=(
                self.cfd_config.domain_size[0] / 2,
                self.cyclone.cylinder_height + self.cyclone.cone_height - self.cyclone.inlet_height / 2,
                0.0
            ),
            inlet_size=(self.cyclone.inlet_width, self.cyclone.inlet_height, self.dx),
            vortex_finder_radius=self.cyclone.vortex_finder_diameter / 2,
            vortex_finder_bottom=self.cyclone.cylinder_height + self.cyclone.cone_height -
                                self.cyclone.vortex_finder_length,
        )

        # Set inlet condition
        self.bc_manager.inlet_condition = InletCondition(
            velocity=(self.cyclone.inlet_velocity, 0.0, 0.0),
            turbulence_intensity=self.cfd_config.turbulence_intensity,
            length_scale=self.cfd_config.turbulence_length_scale,
        )

        # Set outlet condition
        self.bc_manager.outlet_condition = OutletCondition(pressure=0.0)

        # Set wall condition
        self.bc_manager.wall_condition = WallCondition(no_slip=True)

    def _init_particles(self):
        """Initialize particle arrays."""
        n = self.dem_config.num_particles

        self.positions = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.velocities = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.diameters = wp.zeros(n, dtype=float, device=self.device)
        self.masses = wp.zeros(n, dtype=float, device=self.device)
        self.is_active = wp.zeros(n, dtype=wp.int32, device=self.device)

        # Fluid velocity at particle positions
        self.fluid_velocities = wp.zeros(n, dtype=wp.vec3, device=self.device)

        # Initialize particle diameters from material distribution
        diameters_np = self.material.sample_diameters(n)
        masses_np = (np.pi / 6) * diameters_np**3 * self.material.density

        wp.copy(self.diameters, wp.array(diameters_np, dtype=float, device=self.device))
        wp.copy(self.masses, wp.array(masses_np, dtype=float, device=self.device))

        # Track injection
        self._particles_injected = 0
        self._injection_rate = n / self.dem_config.injection_duration

    def _init_coupling_arrays(self):
        """Initialize arrays for two-way coupling."""
        if self.cfd_config.coupling_mode == CouplingMode.TWO_WAY:
            self.momentum_source_x = wp.zeros(
                (self.nx, self.ny, self.nz), dtype=float, device=self.device
            )
            self.momentum_source_y = wp.zeros(
                (self.nx, self.ny, self.nz), dtype=float, device=self.device
            )
            self.momentum_source_z = wp.zeros(
                (self.nx, self.ny, self.nz), dtype=float, device=self.device
            )
            self.particle_count = wp.zeros(
                (self.nx, self.ny, self.nz), dtype=float, device=self.device
            )

    def step_cfd(self):
        """Advance CFD solver by one time step."""
        # Update turbulence and pass eddy viscosity to solver
        if self.turbulence_model is not None:
            if isinstance(self.turbulence_model, KEpsilonModel):
                self.turbulence_model.step(
                    self.cfd_solver.vel_x,
                    self.cfd_solver.vel_y,
                    self.cfd_solver.vel_z,
                    self.cfd_config.dt
                )
                # Pass eddy viscosity to Navier-Stokes solver
                self.cfd_solver.set_eddy_viscosity(
                    self.turbulence_model.get_eddy_viscosity()
                )
            elif isinstance(self.turbulence_model, SmagorinskyModel):
                nu_t = self.turbulence_model.compute_eddy_viscosity(
                    self.cfd_solver.vel_x,
                    self.cfd_solver.vel_y,
                    self.cfd_solver.vel_z,
                )
                # Pass eddy viscosity to Navier-Stokes solver
                self.cfd_solver.set_eddy_viscosity(nu_t)

        # Apply boundary conditions
        self.bc_manager.apply_all(
            self.cfd_solver.vel_x,
            self.cfd_solver.vel_y,
            self.cfd_solver.vel_z,
            self.cfd_solver.pressure,
        )

        # CFD step
        self.cfd_solver.step()

        self._cfd_step += 1

    def step_dem(self):
        """Advance particle system by one time step."""
        dt = self.dem_config.dt
        n = self.dem_config.num_particles

        # Inject particles if still in injection phase
        self._inject_particles()

        # Interpolate fluid velocity to particles
        wp.launch(
            kernel=interpolate_velocity_to_particles,
            dim=n,
            inputs=[
                self.positions,
                self.is_active,
                self.cfd_solver.vel_x,
                self.cfd_solver.vel_y,
                self.cfd_solver.vel_z,
                self.fluid_velocities,
                self.domain_origin,
                self.dx, self.dy, self.dz,
                self.nx, self.ny, self.nz,
            ],
            device=self.device
        )

        # Update particle velocities and positions (simplified - full implementation
        # would include drag, gravity, wall collisions)
        self._update_particles(dt)

        # Two-way coupling: compute momentum source
        if self.cfd_config.coupling_mode == CouplingMode.TWO_WAY:
            self._compute_momentum_coupling()

        self._dem_step += 1
        self._time += dt

    def _inject_particles(self):
        """Inject particles at inlet."""
        if self._time >= self.dem_config.injection_duration:
            return

        # Number to inject this step
        target = int(self._injection_rate * self._time)
        to_inject = min(target - self._particles_injected,
                       self.dem_config.num_particles - self._particles_injected)

        if to_inject <= 0:
            return

        # Set injection positions (at inlet)
        positions_np = self.positions.numpy()
        velocities_np = self.velocities.numpy()
        is_active_np = self.is_active.numpy()

        inlet_x = self.cfd_config.domain_size[0] / 2
        inlet_y = self.cyclone.cylinder_height + self.cyclone.cone_height - self.cyclone.inlet_height / 2
        inlet_z = self.dx  # Just inside domain

        for i in range(self._particles_injected, self._particles_injected + to_inject):
            # Random position within inlet
            positions_np[i] = [
                inlet_x + (np.random.random() - 0.5) * self.cyclone.inlet_width,
                inlet_y + (np.random.random() - 0.5) * self.cyclone.inlet_height,
                inlet_z + np.random.random() * self.dx,
            ]
            velocities_np[i] = [self.cyclone.inlet_velocity, 0.0, 0.0]
            is_active_np[i] = 1

        wp.copy(self.positions, wp.array(positions_np, dtype=wp.vec3, device=self.device))
        wp.copy(self.velocities, wp.array(velocities_np, dtype=wp.vec3, device=self.device))
        wp.copy(self.is_active, wp.array(is_active_np, dtype=wp.int32, device=self.device))

        self._particles_injected += to_inject

    def _update_particles(self, dt: float):
        """Update particle velocities and positions."""
        # This is a simplified update - the full simulator has more detailed physics
        # For now, use drag from fluid velocity difference

        positions_np = self.positions.numpy()
        velocities_np = self.velocities.numpy()
        is_active_np = self.is_active.numpy()
        diameters_np = self.diameters.numpy()
        masses_np = self.masses.numpy()
        fluid_vel_np = self.fluid_velocities.numpy()

        g = 9.81
        rho_f = self.cfd_config.density
        mu_f = self.cfd_config.density * self.cfd_config.kinematic_viscosity

        for i in range(len(positions_np)):
            if is_active_np[i] != 1:
                continue

            pos = positions_np[i]
            vel = velocities_np[i]
            d = diameters_np[i]
            m = masses_np[i]
            vel_f = fluid_vel_np[i]

            # Relative velocity
            vel_rel = vel_f - vel
            vel_rel_mag = np.linalg.norm(vel_rel)

            if vel_rel_mag > 1e-10:
                # Reynolds number
                Re_p = rho_f * vel_rel_mag * d / mu_f

                # Drag coefficient
                if Re_p < 1000:
                    C_D = (24.0 / Re_p) * (1.0 + 0.15 * Re_p**0.687)
                else:
                    C_D = 0.44

                # Drag force
                A = 0.25 * np.pi * d**2
                F_drag = 0.5 * C_D * rho_f * A * vel_rel_mag * vel_rel

                # Acceleration
                a_drag = F_drag / m
            else:
                a_drag = np.zeros(3)

            # Gravity
            a_grav = np.array([0.0, -g, 0.0])

            # Total acceleration
            a_total = a_drag + a_grav

            # Update velocity and position
            velocities_np[i] = vel + a_total * dt
            positions_np[i] = pos + velocities_np[i] * dt

            # Simple boundary check (mark as collected/escaped)
            r = np.sqrt(pos[0]**2 + pos[2]**2)
            if pos[1] < 0:
                is_active_np[i] = -1  # Collected
            elif pos[1] > self.cyclone.cylinder_height + self.cyclone.cone_height:
                if r < self.cyclone.vortex_finder_diameter / 2:
                    is_active_np[i] = -2  # Escaped

        wp.copy(self.positions, wp.array(positions_np, dtype=wp.vec3, device=self.device))
        wp.copy(self.velocities, wp.array(velocities_np, dtype=wp.vec3, device=self.device))
        wp.copy(self.is_active, wp.array(is_active_np, dtype=wp.int32, device=self.device))

    def _compute_momentum_coupling(self):
        """Compute and apply momentum source from particles to fluid."""
        n = self.dem_config.num_particles

        # Reset momentum arrays
        self.momentum_source_x.zero_()
        self.momentum_source_y.zero_()
        self.momentum_source_z.zero_()
        self.particle_count.zero_()

        cell_volume = self.dx * self.dy * self.dz

        # Compute particle momentum source
        wp.launch(
            kernel=compute_particle_momentum_source,
            dim=n,
            inputs=[
                self.positions,
                self.velocities,
                self.diameters,
                self.is_active,
                self.fluid_velocities,
                self.momentum_source_x,
                self.momentum_source_y,
                self.momentum_source_z,
                self.particle_count,
                self.domain_origin,
                self.dx, self.dy, self.dz,
                self.nx, self.ny, self.nz,
                self.cfd_config.density,
                self.cfd_config.density * self.cfd_config.kinematic_viscosity,
                cell_volume,
            ],
            device=self.device
        )

        # Apply to fluid velocity
        wp.launch(
            kernel=apply_momentum_source_to_velocity,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                self.cfd_solver.vel_x,
                self.cfd_solver.vel_y,
                self.cfd_solver.vel_z,
                self.momentum_source_x,
                self.momentum_source_y,
                self.momentum_source_z,
                self.cfd_config.dt,
                self.cfd_config.density,
                self.nx, self.ny, self.nz,
            ],
            device=self.device
        )

    def step(self):
        """
        Advance coupled simulation by one DEM time step.

        Performs multiple CFD substeps per DEM step for stability.
        """
        # CFD substeps
        for _ in range(self.cfd_config.cfd_substeps):
            self.step_cfd()

        # DEM step
        self.step_dem()

    def run(
        self,
        duration: float,
        progress_callback: Optional[Callable] = None
    ):
        """
        Run coupled simulation for specified duration.

        Args:
            duration: Simulation duration [s]
            progress_callback: Optional callback(current_time, total_time)
        """
        while self._time < duration:
            self.step()

            if progress_callback and self._dem_step % 100 == 0:
                progress_callback(self._time, duration)

    def get_results(self) -> dict:
        """Get simulation results."""
        is_active_np = self.is_active.numpy()

        collected = np.sum(is_active_np == -1)
        escaped = np.sum(is_active_np == -2)
        active = np.sum(is_active_np == 1)
        total = collected + escaped

        efficiency = collected / total if total > 0 else 0.0

        return {
            'time': self._time,
            'cfd_steps': self._cfd_step,
            'dem_steps': self._dem_step,
            'particles_injected': self._particles_injected,
            'particles_collected': int(collected),
            'particles_escaped': int(escaped),
            'particles_active': int(active),
            'collection_efficiency': efficiency,
            'positions': self.positions.numpy(),
            'velocities': self.velocities.numpy(),
            'diameters': self.diameters.numpy(),
            'is_active': is_active_np,
        }

    @property
    def time(self) -> float:
        """Current simulation time."""
        return self._time
