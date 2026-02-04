"""
Particle-wall collision handling for cyclone air classifier.

Implements collision detection and response for particles interacting
with cyclone walls using both analytical SDF and mesh-based approaches.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np
import warp as wp

from ...utils.constants import PI, NumericalConstants


@dataclass
class WallCollisionParams:
    """Parameters for particle-wall collisions."""

    restitution_coefficient: float = 0.8    # [-] Normal restitution (0-1)
    friction_coefficient: float = 0.3       # [-] Tangential friction
    wall_roughness: float = 0.0             # [-] Surface roughness factor

    # Collision detection
    collision_tolerance: float = 1.0e-6     # [m] Tolerance for collision detection


# =============================================================================
# ANALYTICAL COLLISION DETECTION (SDF-based)
# =============================================================================

@wp.func
def cyclone_sdf_and_normal(
    pos: wp.vec3,
    center: wp.vec3,
    cylinder_radius: float,
    cylinder_height: float,
    cone_height: float,
    cone_bottom_radius: float,
    vf_radius: float,
    vf_bottom_y: float
) -> wp.vec3:
    """
    Compute signed distance and outward normal for cyclone geometry.

    Returns vec3 where:
    - x component: signed distance (negative = inside)
    - y, z components: normal direction (xy plane component, y component)

    Actually returns (distance, normal_r, normal_y) packed as vec3.
    """
    # Local coordinates (y positive going down into cyclone)
    local_x = pos[0] - center[0]
    local_y = center[1] - pos[1]  # Flip so positive is downward
    local_z = pos[2] - center[2]

    # Radial distance from axis
    r = wp.sqrt(local_x * local_x + local_z * local_z)
    eps = 1.0e-10

    total_height = cylinder_height + cone_height

    # Radial unit vector components
    if r > eps:
        nr_x = local_x / r
        nr_z = local_z / r
    else:
        nr_x = 1.0
        nr_z = 0.0

    # Check if inside vortex finder region
    vf_bottom_local = center[1] - vf_bottom_y
    if local_y < vf_bottom_local and r < vf_radius:
        # Inside vortex finder - distance to VF inner wall
        dist = vf_radius - r
        # Normal points inward (toward axis) since we want to keep particle inside VF
        return wp.vec3(dist, -nr_x, -nr_z)

    # Determine local wall radius based on height
    if local_y < 0.0:
        # Above cylinder top
        R_local = cylinder_radius
        dist = -local_y  # Distance to top cap
        return wp.vec3(dist, 0.0, 1.0)  # Normal points up

    elif local_y <= cylinder_height:
        # In cylinder section
        R_local = cylinder_radius
        dist = R_local - r
        return wp.vec3(dist, nr_x, nr_z)

    elif local_y <= total_height:
        # In cone section - interpolate radius
        cone_y = local_y - cylinder_height
        t = cone_y / cone_height
        R_local = cylinder_radius * (1.0 - t) + cone_bottom_radius * t

        # Normal for cone surface
        dr = cylinder_radius - cone_bottom_radius
        slant = wp.sqrt(cone_height * cone_height + dr * dr)

        # Distance to slant surface
        dist_to_slant = (R_local - r) * cone_height / slant

        # Normal direction (radial + downward component)
        n_r = cone_height / slant
        n_y = dr / slant

        return wp.vec3(dist_to_slant, nr_x * n_r, nr_z * n_r)

    else:
        # Below cone bottom
        dist = local_y - total_height
        return wp.vec3(dist, 0.0, -1.0)  # Normal points down


@wp.func
def reflect_velocity(
    vel: wp.vec3,
    normal: wp.vec3,
    restitution: float,
    friction: float
) -> wp.vec3:
    """
    Reflect velocity off a surface with restitution and friction.

    Args:
        vel: Incoming velocity
        normal: Surface normal (pointing away from surface)
        restitution: Normal coefficient of restitution (0-1)
        friction: Tangential friction coefficient

    Returns:
        Reflected velocity
    """
    # Normalize the normal vector
    n_len = wp.length(normal)
    if n_len < 1.0e-10:
        return vel

    n = normal / n_len

    # Decompose velocity into normal and tangential components
    v_n = wp.dot(vel, n)
    v_normal = n * v_n
    v_tangent = vel - v_normal

    # Only reflect if moving into surface
    if v_n >= 0.0:
        return vel

    # Apply restitution to normal component (reverse direction)
    v_normal_new = -restitution * v_normal

    # Apply friction to tangential component
    v_tan_mag = wp.length(v_tangent)
    if v_tan_mag > 1.0e-10:
        # Friction reduces tangential velocity
        v_tangent_new = v_tangent * (1.0 - friction)
    else:
        v_tangent_new = v_tangent

    return v_normal_new + v_tangent_new


@wp.func
def handle_wall_collision_analytical(
    pos: wp.vec3,
    vel: wp.vec3,
    center: wp.vec3,
    cylinder_radius: float,
    cylinder_height: float,
    cone_height: float,
    cone_bottom_radius: float,
    vf_radius: float,
    vf_bottom_y: float,
    restitution: float,
    friction: float,
    particle_radius: float
) -> wp.vec3:
    """
    Handle particle-wall collision using analytical SDF.

    Returns corrected position and modifies velocity.
    This version returns new position; velocity should be handled separately.
    """
    # Get SDF info
    sdf_info = cyclone_sdf_and_normal(
        pos, center, cylinder_radius, cylinder_height,
        cone_height, cone_bottom_radius, vf_radius, vf_bottom_y
    )

    dist = sdf_info[0]

    # Check if penetrating (distance less than particle radius)
    penetration = particle_radius - dist

    if penetration > 0.0:
        # Get normal direction
        local_x = pos[0] - center[0]
        local_z = pos[2] - center[2]
        r = wp.sqrt(local_x * local_x + local_z * local_z)

        if r > 1.0e-10:
            nr_x = local_x / r
            nr_z = local_z / r
        else:
            nr_x = 1.0
            nr_z = 0.0

        # Construct full normal (pointing inward, toward center)
        normal = wp.vec3(-nr_x * sdf_info[1], 0.0, -nr_z * sdf_info[2])
        n_len = wp.length(normal)

        if n_len > 1.0e-10:
            normal = normal / n_len
            # Push particle out of wall
            new_pos = pos + normal * penetration * 1.01
            return new_pos

    return pos


# =============================================================================
# MESH-BASED COLLISION DETECTION
# =============================================================================

@wp.kernel
def detect_wall_collisions_mesh(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    diameters: wp.array(dtype=float),
    is_active: wp.array(dtype=wp.int32),
    mesh: wp.uint64,
    new_positions: wp.array(dtype=wp.vec3),
    new_velocities: wp.array(dtype=wp.vec3),
    collision_flags: wp.array(dtype=wp.int32),
    restitution: float,
    friction: float,
    max_query_dist: float
):
    """
    Detect and handle particle-wall collisions using mesh queries.

    Uses wp.mesh_query_point to find closest point on mesh surface.
    """
    tid = wp.tid()

    if is_active[tid] != 1:
        new_positions[tid] = positions[tid]
        new_velocities[tid] = velocities[tid]
        collision_flags[tid] = 0
        return

    pos = positions[tid]
    vel = velocities[tid]
    particle_radius = diameters[tid] * 0.5

    # Query closest point on mesh
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)

    if wp.mesh_query_point(mesh, pos, max_query_dist, sign, face_index, face_u, face_v):
        # Get closest point and compute distance
        closest = wp.mesh_eval_position(mesh, face_index, face_u, face_v)
        to_closest = closest - pos
        dist = wp.length(to_closest)

        # Check for collision (particle penetrating or very close to surface)
        # sign > 0 means outside mesh, sign < 0 means inside
        penetration = particle_radius - dist * sign

        if penetration > 0.0 or sign < 0.0:
            # Collision detected
            collision_flags[tid] = 1

            # Get surface normal at closest point
            normal = wp.mesh_eval_face_normal(mesh, face_index)

            # Make sure normal points away from surface (outward for outer wall)
            if sign < 0.0:
                normal = -normal

            # Push particle out of surface
            push_dist = penetration + particle_radius * 0.01
            new_pos = pos + normal * push_dist

            # Reflect velocity
            new_vel = reflect_velocity(vel, normal, restitution, friction)

            new_positions[tid] = new_pos
            new_velocities[tid] = new_vel
        else:
            new_positions[tid] = pos
            new_velocities[tid] = vel
            collision_flags[tid] = 0
    else:
        # No mesh found nearby - particle may be outside domain
        new_positions[tid] = pos
        new_velocities[tid] = vel
        collision_flags[tid] = 0


@wp.kernel
def handle_wall_collisions_sdf(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    diameters: wp.array(dtype=float),
    is_active: wp.array(dtype=wp.int32),
    center: wp.vec3,
    cylinder_radius: float,
    cylinder_height: float,
    cone_height: float,
    cone_bottom_radius: float,
    vf_radius: float,
    vf_bottom_y: float,
    restitution: float,
    friction: float
):
    """
    Handle particle-wall collisions using analytical SDF.

    Modifies positions and velocities in place.
    """
    tid = wp.tid()

    if is_active[tid] != 1:
        return

    pos = positions[tid]
    vel = velocities[tid]
    particle_radius = diameters[tid] * 0.5

    # Local coordinates
    local_x = pos[0] - center[0]
    local_y = center[1] - pos[1]  # Positive going down
    local_z = pos[2] - center[2]

    r = wp.sqrt(local_x * local_x + local_z * local_z)
    eps = 1.0e-10

    total_height = cylinder_height + cone_height

    # Radial unit vector
    if r > eps:
        nr_x = local_x / r
        nr_z = local_z / r
    else:
        nr_x = 1.0
        nr_z = 0.0

    new_pos = pos
    new_vel = vel
    collided = False

    # Check vortex finder collision (for particles in upper region)
    if local_y >= 0.0 and local_y < vf_bottom_y and r < vf_radius + particle_radius:
        # Near or inside vortex finder outer wall
        # This would be collision with VF outer surface - particles shouldn't go through
        pass  # VF collision handled differently

    # Get local wall radius
    if local_y <= cylinder_height:
        R_wall = cylinder_radius
    elif local_y <= total_height:
        cone_y = local_y - cylinder_height
        t = cone_y / cone_height
        R_wall = cylinder_radius * (1.0 - t) + cone_bottom_radius * t
    else:
        R_wall = cone_bottom_radius

    # Check outer wall collision
    penetration = r + particle_radius - R_wall

    if penetration > 0.0 and local_y >= 0.0 and local_y <= total_height:
        # Hitting outer wall
        collided = True

        # Normal points inward (toward center)
        normal = wp.vec3(-nr_x, 0.0, -nr_z)

        # For cone, add vertical component to normal
        if local_y > cylinder_height:
            dr = cylinder_radius - cone_bottom_radius
            slant = wp.sqrt(cone_height * cone_height + dr * dr)
            n_y_comp = -dr / slant  # Negative because y is flipped
            n_r_comp = cone_height / slant
            normal = wp.vec3(-nr_x * n_r_comp, n_y_comp, -nr_z * n_r_comp)
            normal = normal / wp.length(normal)

        # Push particle inside
        new_pos = pos + normal * (penetration + particle_radius * 0.01)

        # Reflect velocity
        new_vel = reflect_velocity(vel, normal, restitution, friction)

    # Check top boundary (above cylinder)
    if local_y < -particle_radius:
        # Above top - check if in vortex finder region
        if r > vf_radius:
            # Outside VF region - push down
            collided = True
            normal = wp.vec3(0.0, -1.0, 0.0)
            new_pos = wp.vec3(pos[0], center[1] + particle_radius * 1.01, pos[2])
            new_vel = reflect_velocity(vel, normal, restitution, friction)

    # Check bottom boundary
    if local_y > total_height + particle_radius:
        # Below cone bottom
        if r > cone_bottom_radius:
            # Outside dust outlet - push up
            collided = True
            normal = wp.vec3(0.0, 1.0, 0.0)
            bottom_y = center[1] - total_height
            new_pos = wp.vec3(pos[0], bottom_y + particle_radius * 1.01, pos[2])
            new_vel = reflect_velocity(vel, normal, restitution, friction)

    if collided:
        positions[tid] = new_pos
        velocities[tid] = new_vel


# =============================================================================
# PYTHON HELPER FUNCTIONS
# =============================================================================

def compute_impact_velocity(
    velocity: np.ndarray,
    normal: np.ndarray
) -> Tuple[float, float]:
    """
    Compute normal and tangential impact velocities.

    Args:
        velocity: Particle velocity vector
        normal: Surface normal vector

    Returns:
        Tuple of (normal_velocity, tangential_velocity)
    """
    normal = normal / np.linalg.norm(normal)
    v_n = np.dot(velocity, normal)
    v_normal = v_n * normal
    v_tangent = velocity - v_normal

    return abs(v_n), np.linalg.norm(v_tangent)


def estimate_collision_frequency(
    tangential_velocity: float,
    radial_position: float,
    cyclone_radius: float
) -> float:
    """
    Estimate collision frequency for a particle in swirling flow.

    Args:
        tangential_velocity: Tangential velocity [m/s]
        radial_position: Distance from axis [m]
        cyclone_radius: Cyclone wall radius [m]

    Returns:
        Estimated collisions per second
    """
    if radial_position < 1e-10:
        return 0.0

    # Approximate orbital period
    circumference = 2 * PI * radial_position
    orbital_period = circumference / max(tangential_velocity, 1e-10)

    # Particles near wall collide more frequently
    proximity_factor = radial_position / cyclone_radius

    return proximity_factor / orbital_period


# =============================================================================
# GENERIC WALL COLLISION FUNCTIONS
# Reusable across feed system, ductwork, and classification system
# =============================================================================

@wp.func
def cylindrical_wall_collision(
    pos: wp.vec3,
    vel: wp.vec3,
    center: wp.vec3,
    axis: wp.int32,      # 0=X, 1=Y, 2=Z
    radius: float,
    length: float,
    particle_radius: float,
    restitution: float,
    friction: float
) -> wp.vec3:
    """
    Handle collision with a cylindrical wall.
    
    Reusable for ducts, hoppers, airlocks, screw housings, etc.
    
    Args:
        pos: Particle position
        vel: Particle velocity
        center: Cylinder center position
        axis: Cylinder axis (0=X, 1=Y, 2=Z)
        radius: Inner cylinder radius
        length: Cylinder length (not used for infinite cylinder)
        particle_radius: Particle radius
        restitution: Coefficient of restitution
        friction: Friction coefficient
        
    Returns:
        Corrected position after collision
    """
    # Get position relative to center
    local = pos - center
    
    # Compute radial distance in the plane perpendicular to axis
    if axis == 0:  # X axis
        r_sq = local[1] * local[1] + local[2] * local[2]
    elif axis == 1:  # Y axis
        r_sq = local[0] * local[0] + local[2] * local[2]
    else:  # Z axis
        r_sq = local[0] * local[0] + local[1] * local[1]
    
    r = wp.sqrt(r_sq)
    effective_radius = radius - particle_radius
    
    if r > effective_radius and r > 1e-10:
        # Particle is outside cylinder wall - push it back
        penetration = r - effective_radius
        
        # Radial unit vector (pointing outward)
        if axis == 0:
            n = wp.vec3(0.0, local[1] / r, local[2] / r)
        elif axis == 1:
            n = wp.vec3(local[0] / r, 0.0, local[2] / r)
        else:
            n = wp.vec3(local[0] / r, local[1] / r, 0.0)
        
        # Push particle inside
        new_pos = pos - n * (penetration + particle_radius * 0.01)
        return new_pos
    
    return pos


@wp.func
def conical_wall_collision(
    pos: wp.vec3,
    vel: wp.vec3,
    apex: wp.vec3,
    axis_dir: wp.vec3,    # Unit vector along cone axis
    half_angle: float,     # Half angle of cone [rad]
    height: float,         # Height from apex to base
    particle_radius: float,
    restitution: float,
    friction: float
) -> wp.vec3:
    """
    Handle collision with a conical wall (e.g., hopper cone, cyclone cone).
    
    Args:
        pos: Particle position
        vel: Particle velocity
        apex: Position of cone apex
        axis_dir: Unit vector along cone axis (from apex toward base)
        half_angle: Half angle of cone
        height: Height from apex to base
        particle_radius: Particle radius
        restitution: Coefficient of restitution
        friction: Friction coefficient
        
    Returns:
        Corrected position after collision
    """
    # Vector from apex to particle
    to_particle = pos - apex
    
    # Project onto axis to get axial distance
    axial_dist = wp.dot(to_particle, axis_dir)
    
    # Clamp to valid cone region
    if axial_dist < 0.0 or axial_dist > height:
        return pos
    
    # Radial component (perpendicular to axis)
    axial_vec = axis_dir * axial_dist
    radial_vec = to_particle - axial_vec
    radial_dist = wp.length(radial_vec)
    
    # Cone radius at this axial position
    cone_radius = axial_dist * wp.tan(half_angle)
    effective_radius = cone_radius - particle_radius / wp.cos(half_angle)
    
    if radial_dist > effective_radius and radial_dist > 1e-10:
        # Particle is outside cone wall
        penetration = radial_dist - effective_radius
        
        # Radial unit vector
        radial_unit = radial_vec / radial_dist
        
        # Normal to cone surface (pointing inward)
        # n = -radial * cos(angle) + axis * sin(angle)
        cos_a = wp.cos(half_angle)
        sin_a = wp.sin(half_angle)
        normal = -radial_unit * cos_a + axis_dir * sin_a
        
        # Push particle inside along normal
        new_pos = pos + normal * (penetration + particle_radius * 0.01) / cos_a
        return new_pos
    
    return pos


@wp.func
def rectangular_duct_collision(
    pos: wp.vec3,
    vel: wp.vec3,
    center: wp.vec3,
    axis: wp.int32,       # Duct axis: 0=X, 1=Y, 2=Z
    width: float,         # Width in first perpendicular direction
    height: float,        # Height in second perpendicular direction
    particle_radius: float,
    restitution: float,
    friction: float
) -> wp.vec3:
    """
    Handle collision with a rectangular duct wall.
    
    Args:
        pos: Particle position
        vel: Particle velocity
        center: Duct centerline position
        axis: Duct axis direction (flow direction)
        width: Duct width (perpendicular to axis, in positive coord direction)
        height: Duct height (perpendicular to axis and width)
        particle_radius: Particle radius
        restitution: Coefficient of restitution
        friction: Friction coefficient
        
    Returns:
        Corrected position after collision
    """
    # Get local position relative to centerline
    local = pos - center
    
    # Effective half-dimensions
    half_w = (width - 2.0 * particle_radius) * 0.5
    half_h = (height - 2.0 * particle_radius) * 0.5
    
    new_pos = pos
    
    if axis == 0:  # X axis duct
        # Y-Z cross section
        if local[1] > half_w:
            new_pos = wp.vec3(pos[0], center[1] + half_w, pos[2])
        elif local[1] < -half_w:
            new_pos = wp.vec3(pos[0], center[1] - half_w, pos[2])
        
        if local[2] > half_h:
            new_pos = wp.vec3(new_pos[0], new_pos[1], center[2] + half_h)
        elif local[2] < -half_h:
            new_pos = wp.vec3(new_pos[0], new_pos[1], center[2] - half_h)
            
    elif axis == 1:  # Y axis duct
        # X-Z cross section
        if local[0] > half_w:
            new_pos = wp.vec3(center[0] + half_w, pos[1], pos[2])
        elif local[0] < -half_w:
            new_pos = wp.vec3(center[0] - half_w, pos[1], pos[2])
        
        if local[2] > half_h:
            new_pos = wp.vec3(new_pos[0], new_pos[1], center[2] + half_h)
        elif local[2] < -half_h:
            new_pos = wp.vec3(new_pos[0], new_pos[1], center[2] - half_h)
            
    else:  # Z axis duct
        # X-Y cross section
        if local[0] > half_w:
            new_pos = wp.vec3(center[0] + half_w, pos[1], pos[2])
        elif local[0] < -half_w:
            new_pos = wp.vec3(center[0] - half_w, pos[1], pos[2])
        
        if local[1] > half_h:
            new_pos = wp.vec3(new_pos[0], center[1] + half_h, new_pos[2])
        elif local[1] < -half_h:
            new_pos = wp.vec3(new_pos[0], center[1] - half_h, new_pos[2])
    
    return new_pos


@wp.kernel
def handle_generic_wall_collisions(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    diameters: wp.array(dtype=float),
    is_active: wp.array(dtype=wp.int32),
    # Wall parameters
    wall_type: wp.int32,       # 0=cylinder, 1=cone, 2=rect_duct
    wall_center: wp.vec3,
    wall_axis: wp.int32,       # 0=X, 1=Y, 2=Z
    wall_param1: float,        # radius (cyl/cone) or width (rect)
    wall_param2: float,        # length (cyl) or half_angle (cone) or height (rect)
    wall_param3: float,        # height (cone) or 0 (others)
    restitution: float,
    friction: float
):
    """
    Generic wall collision kernel - handles multiple geometry types.
    
    This kernel can be reused across different simulation contexts.
    """
    tid = wp.tid()
    
    if is_active[tid] != 1:
        return
    
    pos = positions[tid]
    vel = velocities[tid]
    particle_radius = diameters[tid] * 0.5
    
    new_pos = pos
    
    if wall_type == 0:
        # Cylindrical wall
        new_pos = cylindrical_wall_collision(
            pos, vel, wall_center, wall_axis,
            wall_param1, wall_param2, particle_radius,
            restitution, friction
        )
    elif wall_type == 1:
        # Conical wall
        axis_dir = wp.vec3(0.0, 0.0, 0.0)
        if wall_axis == 0:
            axis_dir = wp.vec3(1.0, 0.0, 0.0)
        elif wall_axis == 1:
            axis_dir = wp.vec3(0.0, 1.0, 0.0)
        else:
            axis_dir = wp.vec3(0.0, 0.0, 1.0)
        
        new_pos = conical_wall_collision(
            pos, vel, wall_center, axis_dir,
            wall_param1, wall_param2, wall_param3, particle_radius,
            restitution, friction
        )
    elif wall_type == 2:
        # Rectangular duct
        new_pos = rectangular_duct_collision(
            pos, vel, wall_center, wall_axis,
            wall_param1, wall_param2, particle_radius,
            restitution, friction
        )
    
    # Apply position correction
    if wp.length(new_pos - pos) > 1e-10:
        # Calculate normal from position change
        delta = new_pos - pos
        normal = delta / wp.length(delta)
        
        # Reflect velocity
        new_vel = reflect_velocity(vel, normal, restitution, friction)
        velocities[tid] = new_vel
        positions[tid] = new_pos


# =============================================================================
# WALL COLLISION HANDLER CLASS
# Reusable manager for complex geometries with multiple wall sections
# =============================================================================

@dataclass
class WallSection:
    """Definition of a single wall section."""
    wall_type: int          # 0=cylinder, 1=cone, 2=rect_duct
    center: Tuple[float, float, float]
    axis: int               # 0=X, 1=Y, 2=Z
    param1: float           # radius/width
    param2: float           # length/half_angle/height
    param3: float = 0.0     # height for cone


class WallCollisionHandler:
    """
    Manager for handling wall collisions with complex geometries.
    
    Supports multiple wall sections (cylinders, cones, rectangular ducts)
    that can be combined to represent feed hoppers, ductwork, and cyclones.
    
    Example:
        handler = WallCollisionHandler(restitution=0.3, friction=0.4)
        
        # Add hopper cone
        handler.add_cone(
            apex=(0, 2, 0), axis=1, half_angle=30, height=1.0
        )
        
        # Add cylindrical section
        handler.add_cylinder(
            center=(0, 1.5, 0), axis=1, radius=0.3, length=0.5
        )
        
        # Process collisions
        handler.process(positions, velocities, diameters, is_active, device)
    """
    
    def __init__(
        self,
        restitution: float = 0.3,
        friction: float = 0.4,
        device: str = "cuda"
    ):
        self.restitution = restitution
        self.friction = friction
        self.device = device
        self.wall_sections: List[WallSection] = []
    
    def add_cylinder(
        self,
        center: Tuple[float, float, float],
        axis: int,
        radius: float,
        length: float
    ):
        """Add a cylindrical wall section."""
        self.wall_sections.append(WallSection(
            wall_type=0,
            center=center,
            axis=axis,
            param1=radius,
            param2=length,
        ))
    
    def add_cone(
        self,
        apex: Tuple[float, float, float],
        axis: int,
        half_angle_deg: float,
        height: float
    ):
        """Add a conical wall section."""
        import math
        self.wall_sections.append(WallSection(
            wall_type=1,
            center=apex,
            axis=axis,
            param1=math.radians(half_angle_deg),
            param2=height,
            param3=height,
        ))
    
    def add_rectangular_duct(
        self,
        center: Tuple[float, float, float],
        axis: int,
        width: float,
        height: float
    ):
        """Add a rectangular duct wall section."""
        self.wall_sections.append(WallSection(
            wall_type=2,
            center=center,
            axis=axis,
            param1=width,
            param2=height,
        ))
    
    def process(
        self,
        positions: "wp.array",
        velocities: "wp.array",
        diameters: "wp.array",
        is_active: "wp.array",
    ):
        """
        Process wall collisions for all wall sections.
        
        Args:
            positions: Particle positions (Warp array)
            velocities: Particle velocities (Warp array)
            diameters: Particle diameters (Warp array)
            is_active: Active flags (Warp array)
        """
        n = len(positions)
        
        for section in self.wall_sections:
            wp.launch(
                kernel=handle_generic_wall_collisions,
                dim=n,
                inputs=[
                    positions,
                    velocities,
                    diameters,
                    is_active,
                    section.wall_type,
                    wp.vec3(*section.center),
                    section.axis,
                    section.param1,
                    section.param2,
                    section.param3,
                    self.restitution,
                    self.friction,
                ],
                device=self.device
            )
    
    def clear(self):
        """Remove all wall sections."""
        self.wall_sections.clear()
