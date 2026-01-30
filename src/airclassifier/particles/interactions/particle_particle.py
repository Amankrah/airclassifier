"""
Particle-particle collision handling.

Implements collision detection and response between particles.
This is typically optional for dilute flows in cyclones but can
be important for high particle loading.
"""

from dataclasses import dataclass
import numpy as np
import warp as wp


@dataclass
class ParticleCollisionParams:
    """Parameters for particle-particle collisions."""

    restitution_coefficient: float = 0.9    # [-] Normal restitution
    friction_coefficient: float = 0.2       # [-] Tangential friction
    enable_collisions: bool = False         # Whether to compute collisions

    # Performance settings
    max_neighbors: int = 32                 # Max neighbors to check per particle


@wp.func
def particle_particle_collision(
    pos1: wp.vec3,
    vel1: wp.vec3,
    mass1: float,
    radius1: float,
    pos2: wp.vec3,
    vel2: wp.vec3,
    mass2: float,
    radius2: float,
    restitution: float
) -> wp.vec3:
    """
    Compute velocity change for particle 1 from collision with particle 2.

    Uses simple elastic/inelastic collision model.

    Args:
        pos1, vel1, mass1, radius1: Particle 1 properties
        pos2, vel2, mass2, radius2: Particle 2 properties
        restitution: Coefficient of restitution

    Returns:
        New velocity for particle 1
    """
    # Vector from particle 2 to particle 1
    delta = pos1 - pos2
    dist = wp.length(delta)

    # Check for overlap
    min_dist = radius1 + radius2
    if dist >= min_dist or dist < 1.0e-10:
        return vel1  # No collision

    # Normal direction
    normal = delta / dist

    # Relative velocity
    v_rel = vel1 - vel2

    # Normal component of relative velocity
    v_n = wp.dot(v_rel, normal)

    # Only process if particles approaching
    if v_n >= 0.0:
        return vel1

    # Compute impulse (conservation of momentum with restitution)
    # J = -(1 + e) * v_n / (1/m1 + 1/m2)
    impulse_mag = -(1.0 + restitution) * v_n / (1.0 / mass1 + 1.0 / mass2)

    # Apply impulse to particle 1
    new_vel = vel1 + (impulse_mag / mass1) * normal

    return new_vel


@wp.kernel
def detect_particle_collisions(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    diameters: wp.array(dtype=float),
    masses: wp.array(dtype=float),
    is_active: wp.array(dtype=wp.int32),
    grid: wp.uint64,
    new_velocities: wp.array(dtype=wp.vec3),
    restitution: float,
    search_radius: float
):
    """
    Detect and handle particle-particle collisions using hash grid.

    Args:
        positions: Particle positions
        velocities: Particle velocities
        diameters: Particle diameters
        masses: Particle masses
        is_active: Active flags
        grid: Hash grid for neighbor search
        new_velocities: Output velocities after collisions
        restitution: Coefficient of restitution
        search_radius: Maximum search radius for neighbors
    """
    tid = wp.tid()

    if is_active[tid] != 1:
        new_velocities[tid] = velocities[tid]
        return

    pos1 = positions[tid]
    vel1 = velocities[tid]
    d1 = diameters[tid]
    m1 = masses[tid]
    r1 = d1 * 0.5

    # Query neighbors
    query = wp.hash_grid_query(grid, pos1, search_radius)
    neighbor_idx = int(0)

    accumulated_vel = vel1

    while wp.hash_grid_query_next(query, neighbor_idx):
        if neighbor_idx == tid:
            continue

        if is_active[neighbor_idx] != 1:
            continue

        pos2 = positions[neighbor_idx]
        vel2 = velocities[neighbor_idx]
        d2 = diameters[neighbor_idx]
        m2 = masses[neighbor_idx]
        r2 = d2 * 0.5

        # Check collision
        accumulated_vel = particle_particle_collision(
            pos1, accumulated_vel, m1, r1,
            pos2, vel2, m2, r2,
            restitution
        )

    new_velocities[tid] = accumulated_vel


@wp.kernel
def separate_overlapping_particles(
    positions: wp.array(dtype=wp.vec3),
    diameters: wp.array(dtype=float),
    is_active: wp.array(dtype=wp.int32),
    grid: wp.uint64,
    new_positions: wp.array(dtype=wp.vec3),
    search_radius: float
):
    """
    Separate overlapping particles by pushing them apart.

    This is a simple position correction to prevent particles
    from occupying the same space.
    """
    tid = wp.tid()

    if is_active[tid] != 1:
        new_positions[tid] = positions[tid]
        return

    pos1 = positions[tid]
    r1 = diameters[tid] * 0.5

    # Query neighbors
    query = wp.hash_grid_query(grid, pos1, search_radius)
    neighbor_idx = int(0)

    correction = wp.vec3(0.0, 0.0, 0.0)
    num_overlaps = 0

    while wp.hash_grid_query_next(query, neighbor_idx):
        if neighbor_idx == tid:
            continue

        if is_active[neighbor_idx] != 1:
            continue

        pos2 = positions[neighbor_idx]
        r2 = diameters[neighbor_idx] * 0.5

        delta = pos1 - pos2
        dist = wp.length(delta)
        min_dist = r1 + r2

        if dist < min_dist and dist > 1.0e-10:
            # Overlap detected - push apart
            overlap = min_dist - dist
            normal = delta / dist
            correction = correction + normal * overlap * 0.5
            num_overlaps += 1

    if num_overlaps > 0:
        new_positions[tid] = pos1 + correction
    else:
        new_positions[tid] = pos1


class ParticleCollisionHandler:
    """
    Manages particle-particle collision detection and response.

    Uses a hash grid for efficient neighbor queries.
    """

    def __init__(
        self,
        params: ParticleCollisionParams,
        max_particles: int,
        device: str = "cuda"
    ):
        """
        Initialize collision handler.

        Args:
            params: Collision parameters
            max_particles: Maximum number of particles
            device: Warp device
        """
        self.params = params
        self.max_particles = max_particles
        self.device = device

        # Create hash grid
        self.grid = None
        self._grid_built = False

    def build_grid(
        self,
        positions: wp.array,
        search_radius: float
    ):
        """
        Build hash grid for neighbor queries.

        Args:
            positions: Particle positions array
            search_radius: Search radius for queries
        """
        if self.grid is None:
            # Estimate grid size based on search radius
            grid_dim = 128
            self.grid = wp.HashGrid(grid_dim, grid_dim, grid_dim, device=self.device)

        self.grid.build(positions, search_radius)
        self._grid_built = True

    def process_collisions(
        self,
        positions: wp.array,
        velocities: wp.array,
        diameters: wp.array,
        masses: wp.array,
        is_active: wp.array,
        search_radius: float
    ) -> wp.array:
        """
        Process particle-particle collisions.

        Args:
            positions: Particle positions
            velocities: Particle velocities
            diameters: Particle diameters
            masses: Particle masses
            is_active: Active flags
            search_radius: Maximum interaction distance

        Returns:
            Array of new velocities
        """
        if not self.params.enable_collisions:
            return velocities

        n = len(positions)

        # Build grid
        self.build_grid(positions, search_radius)

        # Allocate output
        new_velocities = wp.zeros(n, dtype=wp.vec3, device=self.device)

        # Launch kernel
        wp.launch(
            kernel=detect_particle_collisions,
            dim=n,
            inputs=[
                positions,
                velocities,
                diameters,
                masses,
                is_active,
                self.grid.id,
                new_velocities,
                self.params.restitution_coefficient,
                search_radius
            ],
            device=self.device
        )

        return new_velocities
