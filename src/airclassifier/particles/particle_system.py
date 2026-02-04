"""
GPU-accelerated particle system using NVIDIA Warp.

Provides comprehensive particle simulation for air classification of food powders,
including protein separation from yellow peas, faba beans, and oat flour.

Features:
- GPU-accelerated particle dynamics with Warp kernels
- Hash grid spatial partitioning for neighbor queries
- Multiple drag models (Stokes, Schiller-Naumann, Haider-Levenspiel)
- Particle-particle and particle-wall interactions
- Support for polydisperse particle populations
- Real-time visualization integration
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union
from enum import Enum
import numpy as np
import warp as wp

from .material import (
    ParticleMaterial,
    WarpMaterialProps,
    material_to_warp,
    particle_volume,
    particle_mass,
)
from ..utils.constants import (
    PI, 
    GRAVITY, 
    AirProperties,
    FoodPowderComposition,
)


# =============================================================================
# ENUMS AND CONFIGURATION
# =============================================================================

class ParticleType(Enum):
    """Type of particle in food powder classification."""
    WHOLE = 0           # Unseparated whole flour
    PROTEIN = 1         # Protein-rich fraction (fines)
    STARCH = 2          # Starch-rich fraction (coarse)
    FIBER = 3           # Fiber/bran fraction
    OTHER = 4           # Other components


class IntegrationMethod(Enum):
    """Time integration method."""
    EULER = "euler"
    SYMPLECTIC_EULER = "symplectic_euler"
    VELOCITY_VERLET = "velocity_verlet"


class CollisionState(Enum):
    """Particle collision/collection state."""
    ACTIVE = 1          # Particle is active in simulation
    COLLECTED_FINES = -1     # Collected as fines (protein-rich)
    COLLECTED_COARSE = -2    # Collected as coarse (starch-rich)
    COLLECTED_CYCLONE_1 = -3
    COLLECTED_CYCLONE_2 = -4
    COLLECTED_CYCLONE_3 = -5
    COLLECTED_BAG_FILTER = -6
    WALL_COLLISION = -7      # Stuck on wall
    OUT_OF_BOUNDS = -8       # Left simulation domain


@dataclass
class ParticleSystemConfig:
    """Configuration for the particle system."""
    
    # Capacity
    max_particles: int = 100000
    
    # Physics
    gravity: float = GRAVITY
    include_gravity: bool = True
    include_drag: bool = True
    include_centrifugal: bool = True
    include_virtual_mass: bool = False  # For dense suspensions
    
    # Fluid properties
    fluid_density: float = AirProperties.DENSITY
    fluid_viscosity: float = AirProperties.DYNAMIC_VISCOSITY
    
    # Interaction parameters
    enable_particle_collisions: bool = False  # Expensive, usually not needed
    collision_radius_multiplier: float = 1.2
    restitution_coefficient: float = 0.3
    friction_coefficient: float = 0.5
    
    # Wall collision
    wall_restitution: float = 0.3
    wall_friction: float = 0.5
    
    # Hash grid for neighbor search
    hash_grid_dim: int = 128
    neighbor_search_radius: float = 0.001  # [m]
    
    # Numerical
    integration_method: IntegrationMethod = IntegrationMethod.SYMPLECTIC_EULER
    
    # Device
    device: str = "cuda"


# =============================================================================
# WARP STRUCTURES
# =============================================================================

@wp.struct
class WarpParticleData:
    """GPU-side particle data structure."""
    position: wp.vec3
    velocity: wp.vec3
    diameter: float
    density: float
    mass: float
    particle_type: int      # ParticleType enum value
    state: int              # CollisionState enum value
    age: float              # Time since injection


@wp.struct
class WarpFluidParams:
    """Fluid properties for drag calculations."""
    density: float
    viscosity: float
    gravity: float


@wp.struct
class WarpDomainBounds:
    """Simulation domain boundaries."""
    min_bound: wp.vec3
    max_bound: wp.vec3
    is_periodic: wp.vec3i  # 1 for periodic, 0 for reflective


# =============================================================================
# WARP FUNCTIONS
# =============================================================================

@wp.func
def compute_particle_reynolds(
    v_rel: wp.vec3,
    diameter: float,
    rho_f: float,
    mu_f: float
) -> float:
    """Compute particle Reynolds number."""
    v_mag = wp.length(v_rel)
    if v_mag < 1.0e-12:
        return 0.0
    return rho_f * v_mag * diameter / mu_f


@wp.func
def drag_coefficient_schiller_naumann(Re_p: float) -> float:
    """Schiller-Naumann drag coefficient."""
    if Re_p < 1.0e-10:
        return 2.4e11  # Large value for numerical stability
    if Re_p < 1000.0:
        return (24.0 / Re_p) * (1.0 + 0.15 * wp.pow(Re_p, 0.687))
    return 0.44


@wp.func
def drag_coefficient_haider_levenspiel(Re_p: float, sphericity: float) -> float:
    """
    Haider-Levenspiel drag coefficient for non-spherical particles.
    
    Valid for 0.01 < Re_p < 100000 and 0.25 < sphericity < 1.0
    """
    if Re_p < 1.0e-10:
        return 2.4e11
    
    # Coefficients depend on sphericity
    A = wp.exp(2.3288 - 6.4581 * sphericity + 2.4486 * sphericity * sphericity)
    B = 0.0964 + 0.5565 * sphericity
    C = wp.exp(4.905 - 13.8944 * sphericity + 18.4222 * sphericity * sphericity 
               - 10.2599 * sphericity * sphericity * sphericity)
    D = wp.exp(1.4681 + 12.2584 * sphericity - 20.7322 * sphericity * sphericity 
               + 15.8855 * sphericity * sphericity * sphericity)
    
    Cd = (24.0 / Re_p) * (1.0 + A * wp.pow(Re_p, B)) + C / (1.0 + D / Re_p)
    return Cd


@wp.func
def compute_drag_acceleration(
    v_particle: wp.vec3,
    v_fluid: wp.vec3,
    diameter: float,
    rho_p: float,
    rho_f: float,
    mu_f: float,
    sphericity: float
) -> wp.vec3:
    """
    Compute drag acceleration on a particle.
    
    Uses Haider-Levenspiel for non-spherical particles.
    """
    v_rel = v_fluid - v_particle
    v_rel_mag = wp.length(v_rel)
    
    if v_rel_mag < 1.0e-12 or diameter < 1.0e-12:
        return wp.vec3(0.0, 0.0, 0.0)
    
    Re_p = rho_f * v_rel_mag * diameter / mu_f
    
    # Use sphericity-dependent drag
    if sphericity > 0.99:
        Cd = drag_coefficient_schiller_naumann(Re_p)
    else:
        Cd = drag_coefficient_haider_levenspiel(Re_p, sphericity)
    
    # Drag force: F = 0.5 * Cd * rho_f * A * |v_rel|^2
    # Acceleration: a = F / m = (3/4) * Cd * rho_f * |v_rel|^2 / (rho_p * d)
    A_p = 0.25 * 3.141592653589793 * diameter * diameter
    volume = (3.141592653589793 / 6.0) * diameter * diameter * diameter
    mass = rho_p * volume
    
    F_drag_mag = 0.5 * Cd * rho_f * A_p * v_rel_mag * v_rel_mag
    a_drag_mag = F_drag_mag / mass
    
    # Direction is along relative velocity
    v_rel_unit = v_rel / v_rel_mag
    return v_rel_unit * a_drag_mag


@wp.func
def compute_centrifugal_acceleration(
    pos: wp.vec3,
    vel: wp.vec3,
    axis_center: wp.vec3
) -> wp.vec3:
    """
    Compute centrifugal acceleration for swirling flow.
    
    Assumes Y-axis aligned rotation (vertical cyclone).
    """
    # Radial distance in XZ plane
    dx = pos[0] - axis_center[0]
    dz = pos[2] - axis_center[2]
    r = wp.sqrt(dx * dx + dz * dz)
    
    if r < 1.0e-10:
        return wp.vec3(0.0, 0.0, 0.0)
    
    # Radial unit vector
    r_unit_x = dx / r
    r_unit_z = dz / r
    
    # Tangential velocity (perpendicular to radial in XZ plane)
    v_rad = vel[0] * r_unit_x + vel[2] * r_unit_z
    v_tan_x = vel[0] - v_rad * r_unit_x
    v_tan_z = vel[2] - v_rad * r_unit_z
    v_tan_sq = v_tan_x * v_tan_x + v_tan_z * v_tan_z
    
    # Centrifugal acceleration = v_tan^2 / r (radially outward)
    a_cent = v_tan_sq / r
    
    return wp.vec3(a_cent * r_unit_x, 0.0, a_cent * r_unit_z)


# =============================================================================
# WARP KERNELS
# =============================================================================

@wp.kernel
def initialize_particles_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    diameters: wp.array(dtype=float),
    densities: wp.array(dtype=float),
    masses: wp.array(dtype=float),
    particle_types: wp.array(dtype=wp.int32),
    states: wp.array(dtype=wp.int32),
    ages: wp.array(dtype=float),
    # Input data
    init_positions: wp.array(dtype=wp.vec3),
    init_velocities: wp.array(dtype=wp.vec3),
    init_diameters: wp.array(dtype=float),
    init_densities: wp.array(dtype=float),
    init_types: wp.array(dtype=wp.int32),
    start_index: int,
    count: int,
):
    """Initialize particle data on GPU."""
    tid = wp.tid()
    
    if tid >= count:
        return
    
    idx = start_index + tid
    
    positions[idx] = init_positions[tid]
    velocities[idx] = init_velocities[tid]
    diameters[idx] = init_diameters[tid]
    densities[idx] = init_densities[tid]
    
    d = init_diameters[tid]
    rho = init_densities[tid]
    masses[idx] = (3.141592653589793 / 6.0) * d * d * d * rho
    
    particle_types[idx] = init_types[tid]
    states[idx] = 1  # ACTIVE
    ages[idx] = 0.0


@wp.kernel
def compute_accelerations_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    diameters: wp.array(dtype=float),
    densities: wp.array(dtype=float),
    masses: wp.array(dtype=float),
    states: wp.array(dtype=wp.int32),
    accelerations: wp.array(dtype=wp.vec3),
    fluid_velocities: wp.array(dtype=wp.vec3),
    # Parameters
    rho_f: float,
    mu_f: float,
    gravity: float,
    sphericity: float,
    include_gravity: bool,
    include_drag: bool,
    include_centrifugal: bool,
    axis_center: wp.vec3,
):
    """Compute total acceleration for each particle."""
    tid = wp.tid()
    
    if states[tid] != 1:  # Not active
        accelerations[tid] = wp.vec3(0.0, 0.0, 0.0)
        return
    
    pos = positions[tid]
    vel = velocities[tid]
    d = diameters[tid]
    rho_p = densities[tid]
    v_f = fluid_velocities[tid]
    
    acc = wp.vec3(0.0, 0.0, 0.0)
    
    # Gravity (with buoyancy)
    if include_gravity:
        buoyancy_factor = 1.0 - rho_f / rho_p
        acc = acc + wp.vec3(0.0, -gravity * buoyancy_factor, 0.0)
    
    # Drag force
    if include_drag:
        a_drag = compute_drag_acceleration(vel, v_f, d, rho_p, rho_f, mu_f, sphericity)
        acc = acc + a_drag
    
    # Centrifugal force (apparent force in swirling flow)
    if include_centrifugal:
        a_cent = compute_centrifugal_acceleration(pos, vel, axis_center)
        acc = acc + a_cent
    
    accelerations[tid] = acc


@wp.kernel
def integrate_euler_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    accelerations: wp.array(dtype=wp.vec3),
    states: wp.array(dtype=wp.int32),
    ages: wp.array(dtype=float),
    dt: float,
):
    """Euler integration."""
    tid = wp.tid()
    
    if states[tid] != 1:
        return
    
    vel = velocities[tid]
    acc = accelerations[tid]
    pos = positions[tid]
    
    new_vel = vel + acc * dt
    new_pos = pos + new_vel * dt
    
    velocities[tid] = new_vel
    positions[tid] = new_pos
    ages[tid] = ages[tid] + dt


@wp.kernel
def integrate_symplectic_euler_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    accelerations: wp.array(dtype=wp.vec3),
    states: wp.array(dtype=wp.int32),
    ages: wp.array(dtype=float),
    dt: float,
):
    """Symplectic Euler integration (better energy conservation)."""
    tid = wp.tid()
    
    if states[tid] != 1:
        return
    
    vel = velocities[tid]
    acc = accelerations[tid]
    pos = positions[tid]
    
    # Update velocity first
    new_vel = vel + acc * dt
    # Then position with new velocity
    new_pos = pos + new_vel * dt
    
    velocities[tid] = new_vel
    positions[tid] = new_pos
    ages[tid] = ages[tid] + dt


@wp.kernel
def apply_boundary_conditions_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    states: wp.array(dtype=wp.int32),
    bounds_min: wp.vec3,
    bounds_max: wp.vec3,
    restitution: float,
):
    """Apply reflective boundary conditions."""
    tid = wp.tid()
    
    if states[tid] != 1:
        return
    
    pos = positions[tid]
    vel = velocities[tid]
    
    # X boundaries
    if pos[0] < bounds_min[0]:
        pos = wp.vec3(bounds_min[0], pos[1], pos[2])
        vel = wp.vec3(-vel[0] * restitution, vel[1], vel[2])
    elif pos[0] > bounds_max[0]:
        pos = wp.vec3(bounds_max[0], pos[1], pos[2])
        vel = wp.vec3(-vel[0] * restitution, vel[1], vel[2])
    
    # Y boundaries
    if pos[1] < bounds_min[1]:
        pos = wp.vec3(pos[0], bounds_min[1], pos[2])
        vel = wp.vec3(vel[0], -vel[1] * restitution, vel[2])
    elif pos[1] > bounds_max[1]:
        pos = wp.vec3(pos[0], bounds_max[1], pos[2])
        vel = wp.vec3(vel[0], -vel[1] * restitution, vel[2])
    
    # Z boundaries
    if pos[2] < bounds_min[2]:
        pos = wp.vec3(pos[0], pos[1], bounds_min[2])
        vel = wp.vec3(vel[0], vel[1], -vel[2] * restitution)
    elif pos[2] > bounds_max[2]:
        pos = wp.vec3(pos[0], pos[1], bounds_max[2])
        vel = wp.vec3(vel[0], vel[1], -vel[2] * restitution)
    
    positions[tid] = pos
    velocities[tid] = vel


@wp.kernel
def check_collection_zones_kernel(
    positions: wp.array(dtype=wp.vec3),
    diameters: wp.array(dtype=float),
    states: wp.array(dtype=wp.int32),
    particle_types: wp.array(dtype=wp.int32),
    # Collection zone definitions
    fines_y_min: float,           # Y position above which = fines
    coarse_y_max: float,          # Y position below which = coarse
    cyclone_center: wp.vec3,      # Cyclone axis center
    cyclone_bottom_y: float,      # Y position of cyclone collection
    cyclone_radius: float,        # Radius for cyclone collection
):
    """Check if particles have entered collection zones."""
    tid = wp.tid()
    
    if states[tid] != 1:  # Not active
        return
    
    pos = positions[tid]
    
    # Fines collection (top exit - protein rich)
    if pos[1] > fines_y_min:
        states[tid] = -1  # COLLECTED_FINES
        return
    
    # Coarse collection (bottom - starch rich)
    if pos[1] < coarse_y_max:
        states[tid] = -2  # COLLECTED_COARSE
        return
    
    # Cyclone collection
    dx = pos[0] - cyclone_center[0]
    dz = pos[2] - cyclone_center[2]
    r = wp.sqrt(dx * dx + dz * dz)
    
    if pos[1] < cyclone_bottom_y and r < cyclone_radius:
        states[tid] = -3  # COLLECTED_CYCLONE_1


@wp.kernel
def compute_rankine_vortex_field_kernel(
    positions: wp.array(dtype=wp.vec3),
    fluid_velocities: wp.array(dtype=wp.vec3),
    states: wp.array(dtype=wp.int32),
    # Vortex parameters
    vortex_center: wp.vec3,
    vortex_core_radius: float,
    max_tangential_velocity: float,
    axial_velocity_up: float,
    axial_velocity_down: float,
    radial_velocity: float,
):
    """
    Compute Rankine vortex flow field at particle positions.
    
    Used for analytical cyclone flow modeling.
    """
    tid = wp.tid()
    
    if states[tid] != 1:
        fluid_velocities[tid] = wp.vec3(0.0, 0.0, 0.0)
        return
    
    pos = positions[tid]
    
    # Radial distance from vortex axis
    dx = pos[0] - vortex_center[0]
    dz = pos[2] - vortex_center[2]
    r = wp.sqrt(dx * dx + dz * dz)
    
    if r < 1.0e-10:
        # At center, upward axial flow
        fluid_velocities[tid] = wp.vec3(0.0, axial_velocity_up, 0.0)
        return
    
    # Unit vectors
    r_hat_x = dx / r
    r_hat_z = dz / r
    
    # Tangential direction (counterclockwise when viewed from above)
    t_hat_x = -r_hat_z
    t_hat_z = r_hat_x
    
    # Tangential velocity (Rankine vortex)
    if r < vortex_core_radius:
        # Solid body rotation in core
        v_tan = max_tangential_velocity * (r / vortex_core_radius)
    else:
        # Free vortex outside core
        v_tan = max_tangential_velocity * vortex_core_radius / r
    
    # Axial velocity (up in center, down near walls)
    r_ratio = r / vortex_core_radius
    if r_ratio < 0.5:
        v_axial = axial_velocity_up
    else:
        v_axial = axial_velocity_up - (axial_velocity_up - axial_velocity_down) * (r_ratio - 0.5) / 0.5
        v_axial = wp.max(v_axial, axial_velocity_down)
    
    # Radial velocity (inward)
    v_rad = radial_velocity
    
    # Compose velocity
    vx = v_tan * t_hat_x + v_rad * r_hat_x
    vz = v_tan * t_hat_z + v_rad * r_hat_z
    
    fluid_velocities[tid] = wp.vec3(vx, v_axial, vz)


@wp.kernel
def compute_statistics_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    diameters: wp.array(dtype=float),
    states: wp.array(dtype=wp.int32),
    particle_types: wp.array(dtype=wp.int32),
    # Output counts
    count_active: wp.array(dtype=wp.int32),
    count_fines: wp.array(dtype=wp.int32),
    count_coarse: wp.array(dtype=wp.int32),
    count_cyclone: wp.array(dtype=wp.int32),
):
    """Compute particle statistics using atomic operations."""
    tid = wp.tid()
    state = states[tid]
    
    if state == 1:
        wp.atomic_add(count_active, 0, 1)
    elif state == -1:
        wp.atomic_add(count_fines, 0, 1)
    elif state == -2:
        wp.atomic_add(count_coarse, 0, 1)
    elif state <= -3 and state >= -6:
        wp.atomic_add(count_cyclone, 0, 1)


# =============================================================================
# PARTICLE SYSTEM CLASS
# =============================================================================

class WarpParticleSystem:
    """
    GPU-accelerated particle system for air classification simulation.
    
    Designed for simulating protein separation from plant-based powders
    (yellow peas, faba beans, oats) through air classifiers.
    
    Example:
        >>> config = ParticleSystemConfig(max_particles=50000)
        >>> system = WarpParticleSystem(config)
        >>> 
        >>> # Create yellow pea flour material
        >>> material = ParticleMaterial.create_food_powder("yellow_pea", "whole")
        >>> 
        >>> # Inject particles
        >>> positions = np.random.randn(1000, 3) * 0.01
        >>> system.inject_particles(positions, material)
        >>> 
        >>> # Simulation loop
        >>> for _ in range(1000):
        ...     system.step(dt=1e-5)
        >>> 
        >>> # Get results
        >>> results = system.get_statistics()
    """
    
    def __init__(self, config: ParticleSystemConfig = None):
        """
        Initialize the particle system.
        
        Args:
            config: System configuration
        """
        self.config = config or ParticleSystemConfig()
        
        # Initialize Warp
        wp.init()
        self.device = self.config.device
        
        # Particle count
        self.num_particles = 0
        self.max_particles = self.config.max_particles
        
        # Allocate GPU arrays
        self._allocate_arrays()
        
        # Hash grid for neighbor search (optional)
        self.hash_grid = None
        if self.config.enable_particle_collisions:
            self._create_hash_grid()
        
        # Domain bounds (can be updated)
        self.bounds_min = wp.vec3(-1.0, -2.0, -1.0)
        self.bounds_max = wp.vec3(1.0, 3.0, 1.0)
        
        # Vortex parameters for cyclone flow
        self.vortex_center = wp.vec3(0.0, 0.0, 0.0)
        self.vortex_core_radius = 0.05
        self.max_tangential_velocity = 15.0
        self.axial_velocity_up = 5.0
        self.axial_velocity_down = -2.0
        self.radial_velocity = -0.5
        
        # Collection zone parameters
        self.fines_y_min = 2.0
        self.coarse_y_max = -1.5
        self.cyclone_bottom_y = -0.5
        self.cyclone_radius = 0.1
        
        # Default sphericity for irregular food particles
        self.sphericity = 0.7
        
        # Statistics arrays
        self._count_active = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._count_fines = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._count_coarse = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._count_cyclone = wp.zeros(1, dtype=wp.int32, device=self.device)
        
        # Time tracking
        self.time = 0.0
        self.step_count = 0
    
    def _allocate_arrays(self):
        """Pre-allocate GPU arrays for particles."""
        n = self.max_particles
        
        # Position and velocity
        self.positions = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.velocities = wp.zeros(n, dtype=wp.vec3, device=self.device)
        
        # Physical properties
        self.diameters = wp.zeros(n, dtype=float, device=self.device)
        self.densities = wp.zeros(n, dtype=float, device=self.device)
        self.masses = wp.zeros(n, dtype=float, device=self.device)
        
        # Classification
        self.particle_types = wp.zeros(n, dtype=wp.int32, device=self.device)
        self.states = wp.zeros(n, dtype=wp.int32, device=self.device)
        self.ages = wp.zeros(n, dtype=float, device=self.device)
        
        # Temporary arrays
        self._accelerations = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self._fluid_velocities = wp.zeros(n, dtype=wp.vec3, device=self.device)
    
    def _create_hash_grid(self):
        """Create hash grid for neighbor search."""
        dim = self.config.hash_grid_dim
        self.hash_grid = wp.HashGrid(dim, dim, dim, device=self.device)
    
    def inject_particles(
        self,
        positions: np.ndarray,
        material: ParticleMaterial,
        velocities: np.ndarray = None,
        particle_type: ParticleType = ParticleType.WHOLE,
    ) -> int:
        """
        Inject new particles into the system.
        
        Args:
            positions: Particle positions (N, 3) in meters
            material: ParticleMaterial with density and size distribution
            velocities: Optional initial velocities (N, 3) in m/s
            particle_type: Type classification for the particles
            
        Returns:
            Number of particles actually injected
        """
        n_inject = len(positions)
        
        # Check capacity
        available = self.max_particles - self.num_particles
        if n_inject > available:
            n_inject = available
            if n_inject == 0:
                return 0
            positions = positions[:n_inject]
        
        # Sample diameters from material distribution
        diameters = material.sample_diameters(n_inject, seed=self.step_count)
        
        # Default velocities
        if velocities is None:
            velocities = np.zeros((n_inject, 3), dtype=np.float32)
        else:
            velocities = velocities[:n_inject]
        
        # Prepare arrays
        pos_wp = wp.array(positions.astype(np.float32), dtype=wp.vec3, device=self.device)
        vel_wp = wp.array(velocities.astype(np.float32), dtype=wp.vec3, device=self.device)
        dia_wp = wp.array(diameters.astype(np.float32), dtype=float, device=self.device)
        den_wp = wp.array(
            np.full(n_inject, material.density, dtype=np.float32),
            dtype=float, device=self.device
        )
        type_wp = wp.array(
            np.full(n_inject, particle_type.value, dtype=np.int32),
            dtype=wp.int32, device=self.device
        )
        
        # Update sphericity based on material
        self.sphericity = material.sphericity
        
        # Launch initialization kernel
        wp.launch(
            kernel=initialize_particles_kernel,
            dim=n_inject,
            inputs=[
                self.positions,
                self.velocities,
                self.diameters,
                self.densities,
                self.masses,
                self.particle_types,
                self.states,
                self.ages,
                pos_wp,
                vel_wp,
                dia_wp,
                den_wp,
                type_wp,
                self.num_particles,
                n_inject,
            ],
            device=self.device
        )
        
        self.num_particles += n_inject
        return n_inject
    
    def inject_mixed_powder(
        self,
        positions: np.ndarray,
        source: str,
        seed: int = 42,
    ) -> int:
        """
        Inject a mixed food powder with realistic composition.
        
        Creates particles with proper fractions of protein, starch, and fiber
        based on the source material composition.
        
        Args:
            positions: Base positions for injection (N, 3)
            source: "yellow_pea", "faba_bean", or "oat"
            seed: Random seed for reproducibility
            
        Returns:
            Total number of particles injected
        """
        rng = np.random.default_rng(seed)
        n_total = len(positions)
        
        # Get composition fractions
        if source.lower() == "yellow_pea":
            f_protein = FoodPowderComposition.YELLOW_PEA_PROTEIN_CONTENT
            f_starch = FoodPowderComposition.YELLOW_PEA_STARCH_CONTENT
            f_fiber = FoodPowderComposition.YELLOW_PEA_FIBER_CONTENT
        elif source.lower() == "faba_bean":
            f_protein = FoodPowderComposition.FABA_BEAN_PROTEIN_CONTENT
            f_starch = FoodPowderComposition.FABA_BEAN_STARCH_CONTENT
            f_fiber = FoodPowderComposition.FABA_BEAN_FIBER_CONTENT
        elif source.lower() == "oat":
            f_protein = FoodPowderComposition.OAT_PROTEIN_CONTENT
            f_starch = FoodPowderComposition.OAT_STARCH_CONTENT
            f_fiber = FoodPowderComposition.OAT_FIBER_CONTENT
        else:
            raise ValueError(f"Unknown source: {source}")
        
        # Normalize fractions
        f_total = f_protein + f_starch + f_fiber
        f_protein /= f_total
        f_starch /= f_total
        f_fiber /= f_total
        
        # Split positions randomly
        n_protein = int(n_total * f_protein)
        n_starch = int(n_total * f_starch)
        n_fiber = n_total - n_protein - n_starch
        
        indices = rng.permutation(n_total)
        
        total_injected = 0
        
        # Inject protein fraction
        if n_protein > 0:
            protein_material = ParticleMaterial.create_food_powder(source, "protein")
            protein_positions = positions[indices[:n_protein]]
            total_injected += self.inject_particles(
                protein_positions, protein_material, particle_type=ParticleType.PROTEIN
            )
        
        # Inject starch fraction
        if n_starch > 0:
            starch_material = ParticleMaterial.create_food_powder(source, "starch")
            starch_positions = positions[indices[n_protein:n_protein+n_starch]]
            total_injected += self.inject_particles(
                starch_positions, starch_material, particle_type=ParticleType.STARCH
            )
        
        # Inject fiber fraction
        if n_fiber > 0:
            if source.lower() == "oat":
                fiber_material = ParticleMaterial.create_food_powder(source, "bran")
            else:
                fiber_material = ParticleMaterial.create_food_powder(source, "fiber")
            fiber_positions = positions[indices[n_protein+n_starch:]]
            total_injected += self.inject_particles(
                fiber_positions, fiber_material, particle_type=ParticleType.FIBER
            )
        
        return total_injected
    
    def set_domain_bounds(
        self,
        min_bound: Tuple[float, float, float],
        max_bound: Tuple[float, float, float]
    ):
        """Set the simulation domain boundaries."""
        self.bounds_min = wp.vec3(min_bound[0], min_bound[1], min_bound[2])
        self.bounds_max = wp.vec3(max_bound[0], max_bound[1], max_bound[2])
    
    def set_vortex_parameters(
        self,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        core_radius: float = 0.05,
        max_tangential_velocity: float = 15.0,
        axial_velocity_up: float = 5.0,
        axial_velocity_down: float = -2.0,
        radial_velocity: float = -0.5,
    ):
        """Configure the Rankine vortex flow field parameters."""
        self.vortex_center = wp.vec3(center[0], center[1], center[2])
        self.vortex_core_radius = core_radius
        self.max_tangential_velocity = max_tangential_velocity
        self.axial_velocity_up = axial_velocity_up
        self.axial_velocity_down = axial_velocity_down
        self.radial_velocity = radial_velocity
    
    def set_collection_zones(
        self,
        fines_y_min: float = 2.0,
        coarse_y_max: float = -1.5,
        cyclone_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        cyclone_bottom_y: float = -0.5,
        cyclone_radius: float = 0.1,
    ):
        """Configure particle collection zones."""
        self.fines_y_min = fines_y_min
        self.coarse_y_max = coarse_y_max
        self.vortex_center = wp.vec3(cyclone_center[0], cyclone_center[1], cyclone_center[2])
        self.cyclone_bottom_y = cyclone_bottom_y
        self.cyclone_radius = cyclone_radius
    
    def compute_fluid_velocities(self):
        """Compute fluid velocities at particle positions using Rankine vortex."""
        if self.num_particles == 0:
            return
        
        wp.launch(
            kernel=compute_rankine_vortex_field_kernel,
            dim=self.num_particles,
            inputs=[
                self.positions,
                self._fluid_velocities,
                self.states,
                self.vortex_center,
                self.vortex_core_radius,
                self.max_tangential_velocity,
                self.axial_velocity_up,
                self.axial_velocity_down,
                self.radial_velocity,
            ],
            device=self.device
        )
    
    def set_fluid_velocities(self, velocities: np.ndarray):
        """Set custom fluid velocities at particle positions."""
        if len(velocities) < self.num_particles:
            raise ValueError("Velocity array too small")
        
        vel_wp = wp.array(
            velocities[:self.num_particles].astype(np.float32),
            dtype=wp.vec3, device=self.device
        )
        wp.copy(self._fluid_velocities, vel_wp, count=self.num_particles)
    
    def step(self, dt: float):
        """
        Advance the simulation by one time step.
        
        Args:
            dt: Time step [s]
        """
        if self.num_particles == 0:
            return
        
        n = self.num_particles
        
        # 1. Compute fluid velocities (Rankine vortex or external)
        self.compute_fluid_velocities()
        
        # 2. Update hash grid if using particle collisions
        if self.hash_grid is not None and self.config.enable_particle_collisions:
            self.hash_grid.build(
                points=self.positions,
                radius=self.config.neighbor_search_radius
            )
        
        # 3. Compute accelerations
        wp.launch(
            kernel=compute_accelerations_kernel,
            dim=n,
            inputs=[
                self.positions,
                self.velocities,
                self.diameters,
                self.densities,
                self.masses,
                self.states,
                self._accelerations,
                self._fluid_velocities,
                self.config.fluid_density,
                self.config.fluid_viscosity,
                self.config.gravity,
                self.sphericity,
                self.config.include_gravity,
                self.config.include_drag,
                self.config.include_centrifugal,
                self.vortex_center,
            ],
            device=self.device
        )
        
        # 4. Integrate
        if self.config.integration_method == IntegrationMethod.SYMPLECTIC_EULER:
            wp.launch(
                kernel=integrate_symplectic_euler_kernel,
                dim=n,
                inputs=[
                    self.positions,
                    self.velocities,
                    self._accelerations,
                    self.states,
                    self.ages,
                    dt,
                ],
                device=self.device
            )
        else:
            wp.launch(
                kernel=integrate_euler_kernel,
                dim=n,
                inputs=[
                    self.positions,
                    self.velocities,
                    self._accelerations,
                    self.states,
                    self.ages,
                    dt,
                ],
                device=self.device
            )
        
        # 5. Apply boundary conditions
        wp.launch(
            kernel=apply_boundary_conditions_kernel,
            dim=n,
            inputs=[
                self.positions,
                self.velocities,
                self.states,
                self.bounds_min,
                self.bounds_max,
                self.config.wall_restitution,
            ],
            device=self.device
        )
        
        # 6. Check collection zones
        wp.launch(
            kernel=check_collection_zones_kernel,
            dim=n,
            inputs=[
                self.positions,
                self.diameters,
                self.states,
                self.particle_types,
                self.fines_y_min,
                self.coarse_y_max,
                self.vortex_center,
                self.cyclone_bottom_y,
                self.cyclone_radius,
            ],
            device=self.device
        )
        
        # Update time
        self.time += dt
        self.step_count += 1
    
    def get_positions(self) -> np.ndarray:
        """Get particle positions as numpy array."""
        return self.positions.numpy()[:self.num_particles]
    
    def get_velocities(self) -> np.ndarray:
        """Get particle velocities as numpy array."""
        return self.velocities.numpy()[:self.num_particles]
    
    def get_diameters(self) -> np.ndarray:
        """Get particle diameters as numpy array."""
        return self.diameters.numpy()[:self.num_particles]
    
    def get_states(self) -> np.ndarray:
        """Get particle states as numpy array."""
        return self.states.numpy()[:self.num_particles]
    
    def get_particle_types(self) -> np.ndarray:
        """Get particle types as numpy array."""
        return self.particle_types.numpy()[:self.num_particles]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute and return simulation statistics.
        
        Returns:
            Dictionary with particle counts, separation efficiency, etc.
        """
        # Reset counters
        self._count_active.zero_()
        self._count_fines.zero_()
        self._count_coarse.zero_()
        self._count_cyclone.zero_()
        
        if self.num_particles > 0:
            wp.launch(
                kernel=compute_statistics_kernel,
                dim=self.num_particles,
                inputs=[
                    self.positions,
                    self.velocities,
                    self.diameters,
                    self.states,
                    self.particle_types,
                    self._count_active,
                    self._count_fines,
                    self._count_coarse,
                    self._count_cyclone,
                ],
                device=self.device
            )
        
        wp.synchronize()
        
        n_active = int(self._count_active.numpy()[0])
        n_fines = int(self._count_fines.numpy()[0])
        n_coarse = int(self._count_coarse.numpy()[0])
        n_cyclone = int(self._count_cyclone.numpy()[0])
        n_collected = n_fines + n_coarse + n_cyclone
        
        # Compute separation efficiency
        efficiency = n_fines / max(1, n_fines + n_coarse) if n_collected > 0 else 0.0
        
        # Get diameter statistics for collected fractions
        states = self.get_states()
        diameters = self.get_diameters()
        types = self.get_particle_types()
        
        fines_mask = states == -1
        coarse_mask = states == -2
        
        fines_d_mean = float(np.mean(diameters[fines_mask])) if np.any(fines_mask) else 0.0
        coarse_d_mean = float(np.mean(diameters[coarse_mask])) if np.any(coarse_mask) else 0.0
        
        # Protein recovery (what fraction of protein went to fines)
        protein_mask = types == ParticleType.PROTEIN.value
        protein_in_fines = np.sum(protein_mask & fines_mask)
        protein_total = np.sum(protein_mask)
        protein_recovery = protein_in_fines / max(1, protein_total)
        
        # Starch rejection (what fraction of starch went to coarse)
        starch_mask = types == ParticleType.STARCH.value
        starch_in_coarse = np.sum(starch_mask & coarse_mask)
        starch_total = np.sum(starch_mask)
        starch_rejection = starch_in_coarse / max(1, starch_total)
        
        return {
            "time": self.time,
            "step_count": self.step_count,
            "total_particles": self.num_particles,
            "active_particles": n_active,
            "collected_fines": n_fines,
            "collected_coarse": n_coarse,
            "collected_cyclone": n_cyclone,
            "separation_efficiency": efficiency,
            "fines_mean_diameter_um": fines_d_mean * 1e6,
            "coarse_mean_diameter_um": coarse_d_mean * 1e6,
            "protein_recovery": protein_recovery,
            "starch_rejection": starch_rejection,
        }
    
    def reset(self):
        """Reset the particle system."""
        self.num_particles = 0
        self.time = 0.0
        self.step_count = 0
        
        # Zero arrays
        self.positions.zero_()
        self.velocities.zero_()
        self.diameters.zero_()
        self.densities.zero_()
        self.masses.zero_()
        self.particle_types.zero_()
        self.states.zero_()
        self.ages.zero_()
    
    def synchronize(self):
        """Synchronize GPU operations."""
        wp.synchronize()
    
    # =========================================================================
    # REUSABLE METHODS FOR PHYSICS SIMULATIONS
    # =========================================================================
    
    def inject_raw_particles(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        diameters: np.ndarray,
        densities: np.ndarray,
        particle_types: np.ndarray = None,
    ) -> int:
        """
        Inject particles with explicit property arrays.
        
        Useful when transferring particles between physics simulations
        (e.g., from feed system to classification system).
        
        Args:
            positions: Particle positions (N, 3) in meters
            velocities: Particle velocities (N, 3) in m/s
            diameters: Particle diameters (N,) in meters
            densities: Particle densities (N,) in kg/m³
            particle_types: Optional particle types (N,) as int
            
        Returns:
            Number of particles actually injected
        """
        n_inject = len(positions)
        
        # Check capacity
        available = self.max_particles - self.num_particles
        if n_inject > available:
            n_inject = available
            if n_inject == 0:
                return 0
        
        # Prepare arrays
        pos_np = positions[:n_inject].astype(np.float32)
        vel_np = velocities[:n_inject].astype(np.float32)
        dia_np = diameters[:n_inject].astype(np.float32)
        den_np = densities[:n_inject].astype(np.float32)
        
        if particle_types is None:
            types_np = np.zeros(n_inject, dtype=np.int32)
        else:
            types_np = particle_types[:n_inject].astype(np.int32)
        
        # Copy to GPU
        pos_wp = wp.array(pos_np, dtype=wp.vec3, device=self.device)
        vel_wp = wp.array(vel_np, dtype=wp.vec3, device=self.device)
        dia_wp = wp.array(dia_np, dtype=float, device=self.device)
        den_wp = wp.array(den_np, dtype=float, device=self.device)
        type_wp = wp.array(types_np, dtype=wp.int32, device=self.device)
        
        # Launch initialization kernel
        wp.launch(
            kernel=initialize_particles_kernel,
            dim=n_inject,
            inputs=[
                self.positions,
                self.velocities,
                self.diameters,
                self.densities,
                self.masses,
                self.particle_types,
                self.states,
                self.ages,
                pos_wp,
                vel_wp,
                dia_wp,
                den_wp,
                type_wp,
                self.num_particles,
                n_inject,
            ],
            device=self.device
        )
        
        self.num_particles += n_inject
        return n_inject
    
    def get_active_particles(self) -> Dict[str, np.ndarray]:
        """
        Get all data for active particles only.
        
        Returns:
            Dictionary with positions, velocities, diameters, densities, types
            for active particles only.
        """
        states = self.get_states()
        active_mask = states == 1
        
        return {
            "positions": self.get_positions()[active_mask],
            "velocities": self.get_velocities()[active_mask],
            "diameters": self.get_diameters()[active_mask],
            "densities": self.densities.numpy()[:self.num_particles][active_mask],
            "types": self.get_particle_types()[active_mask],
            "count": int(np.sum(active_mask)),
        }
    
    def get_exited_particles(self, exit_zone_y: float = None) -> Dict[str, np.ndarray]:
        """
        Get particles that have exited through a boundary.
        
        Args:
            exit_zone_y: Y coordinate threshold for exit (particles below this)
                        If None, uses coarse_y_max
        
        Returns:
            Dictionary with particle data for exited particles
        """
        if exit_zone_y is None:
            exit_zone_y = self.coarse_y_max
        
        positions = self.get_positions()
        states = self.get_states()
        
        # Active particles that crossed exit boundary
        exit_mask = (states == 1) & (positions[:, 1] < exit_zone_y)
        
        return {
            "positions": positions[exit_mask],
            "velocities": self.get_velocities()[exit_mask],
            "diameters": self.get_diameters()[exit_mask],
            "densities": self.densities.numpy()[:self.num_particles][exit_mask],
            "types": self.get_particle_types()[exit_mask],
            "count": int(np.sum(exit_mask)),
        }
    
    def transfer_to_system(
        self,
        target_system: "WarpParticleSystem",
        position_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        exit_zone_y: float = None,
        remove_transferred: bool = True,
    ) -> int:
        """
        Transfer exited particles to another particle system.
        
        This is useful for connecting physics simulations:
        - Feed system → Classification system
        - Classification system → Cyclone system
        
        Args:
            target_system: Target particle system to receive particles
            position_offset: Offset to apply to positions (for coordinate transform)
            exit_zone_y: Y threshold for considering particles as exited
            remove_transferred: Whether to deactivate transferred particles
            
        Returns:
            Number of particles transferred
        """
        exited = self.get_exited_particles(exit_zone_y)
        
        if exited["count"] == 0:
            return 0
        
        # Apply position offset
        new_positions = exited["positions"] + np.array(position_offset)
        
        # Inject into target system
        n_transferred = target_system.inject_raw_particles(
            positions=new_positions,
            velocities=exited["velocities"],
            diameters=exited["diameters"],
            densities=exited["densities"],
            particle_types=exited["types"],
        )
        
        # Deactivate transferred particles in source
        if remove_transferred and n_transferred > 0:
            positions = self.get_positions()
            states = self.states.numpy()
            
            if exit_zone_y is None:
                exit_zone_y = self.coarse_y_max
            
            exit_mask = (states[:self.num_particles] == 1) & (positions[:, 1] < exit_zone_y)
            states[:self.num_particles][exit_mask] = -8  # OUT_OF_BOUNDS
            
            # Copy back to GPU
            wp.copy(self.states, wp.array(states, dtype=wp.int32, device=self.device))
        
        return n_transferred
    
    def update_fluid_velocity_field(
        self,
        velocity_func: callable,
    ):
        """
        Update fluid velocities using a custom function.
        
        Args:
            velocity_func: Function (positions: np.ndarray) -> np.ndarray
                          Takes Nx3 positions, returns Nx3 velocities
        """
        if self.num_particles == 0:
            return
        
        positions = self.get_positions()
        fluid_velocities = velocity_func(positions)
        self.set_fluid_velocities(fluid_velocities)
    
    def get_mass_flow_rate(
        self,
        plane_y: float,
        plane_tolerance: float = 0.01,
        dt: float = 0.001,
    ) -> float:
        """
        Calculate mass flow rate through a horizontal plane.
        
        Args:
            plane_y: Y coordinate of measurement plane
            plane_tolerance: Distance tolerance for considering particle crossing
            dt: Time step used in simulation
            
        Returns:
            Mass flow rate [kg/s]
        """
        positions = self.get_positions()
        velocities = self.get_velocities()
        states = self.get_states()
        
        active_mask = states == 1
        
        # Find particles near the plane with downward velocity
        near_plane = (np.abs(positions[:, 1] - plane_y) < plane_tolerance) & active_mask
        crossing_down = (velocities[:, 1] < 0) & near_plane
        
        if not np.any(crossing_down):
            return 0.0
        
        masses = self.masses.numpy()[:self.num_particles]
        crossing_mass = np.sum(masses[crossing_down])
        
        # Estimate flow rate from crossing frequency
        crossing_velocity = np.mean(np.abs(velocities[crossing_down, 1]))
        crossing_time = plane_tolerance / crossing_velocity if crossing_velocity > 0 else dt
        
        return crossing_mass / crossing_time
    
    def get_particle_data_for_export(self) -> Dict[str, np.ndarray]:
        """
        Get all particle data in a format suitable for export or visualization.
        
        Returns:
            Dictionary with all particle arrays (positions, velocities, etc.)
        """
        return {
            "positions": self.get_positions(),
            "velocities": self.get_velocities(),
            "diameters": self.get_diameters(),
            "densities": self.densities.numpy()[:self.num_particles],
            "masses": self.masses.numpy()[:self.num_particles],
            "types": self.get_particle_types(),
            "states": self.get_states(),
            "ages": self.ages.numpy()[:self.num_particles],
            "num_particles": self.num_particles,
            "time": self.time,
        }
    
    def load_particle_data(self, data: Dict[str, np.ndarray]):
        """
        Load particle data from a dictionary (e.g., from export or another system).
        
        Args:
            data: Dictionary with particle arrays from get_particle_data_for_export()
        """
        self.reset()
        
        n = data.get("num_particles", len(data["positions"]))
        
        self.inject_raw_particles(
            positions=data["positions"][:n],
            velocities=data["velocities"][:n],
            diameters=data["diameters"][:n],
            densities=data["densities"][:n],
            particle_types=data.get("types", None),
        )
        
        # Restore ages if available
        if "ages" in data:
            ages = data["ages"][:n].astype(np.float32)
            wp.copy(
                self.ages,
                wp.array(ages, dtype=float, device=self.device),
                count=n
            )
        
        self.time = data.get("time", 0.0)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_particle_system(
    max_particles: int = 50000,
    device: str = "cuda",
    **kwargs
) -> WarpParticleSystem:
    """
    Create a particle system with default food powder configuration.
    
    Args:
        max_particles: Maximum number of particles
        device: Warp device ("cuda" or "cpu")
        **kwargs: Additional ParticleSystemConfig parameters
        
    Returns:
        Configured WarpParticleSystem
    """
    config = ParticleSystemConfig(
        max_particles=max_particles,
        device=device,
        **kwargs
    )
    return WarpParticleSystem(config)


def create_yellow_pea_simulation(
    num_particles: int = 10000,
    device: str = "cuda",
) -> Tuple[WarpParticleSystem, ParticleMaterial]:
    """
    Create a simulation configured for yellow pea protein separation.
    
    Returns:
        Tuple of (particle_system, material)
    """
    system = create_particle_system(
        max_particles=num_particles * 2,
        device=device,
    )
    material = ParticleMaterial.create_food_powder("yellow_pea", "whole")
    
    return system, material


def create_faba_bean_simulation(
    num_particles: int = 10000,
    device: str = "cuda",
) -> Tuple[WarpParticleSystem, ParticleMaterial]:
    """
    Create a simulation configured for faba bean protein separation.
    
    Returns:
        Tuple of (particle_system, material)
    """
    system = create_particle_system(
        max_particles=num_particles * 2,
        device=device,
    )
    material = ParticleMaterial.create_food_powder("faba_bean", "whole")
    
    return system, material


def create_oat_simulation(
    num_particles: int = 10000,
    device: str = "cuda",
) -> Tuple[WarpParticleSystem, ParticleMaterial]:
    """
    Create a simulation configured for oat protein separation.
    
    Returns:
        Tuple of (particle_system, material)
    """
    system = create_particle_system(
        max_particles=num_particles * 2,
        device=device,
    )
    material = ParticleMaterial.create_food_powder("oat", "whole")
    
    return system, material
