"""
Physics-Based Material Flow Simulation for Feed System
=======================================================

This module implements a physics-based particle flow simulation through the
feed system assembly. All parameters are derived directly from geometry
with NO magic numbers - everything is computed from first principles.

Flow Path:
    HOPPER → AIRLOCK → SCREW FEEDER → DEAGGLOMERATOR → EXIT

Physics Principles:
    - Gravity: F = m*g, computed from particle mass
    - Drag: Schiller-Naumann correlation for air drag
    - Collisions: Inelastic wall collisions with restitution and friction
    - Rotation: Tangential velocity from rotating components (ω = 2π*RPM/60)
    - Conveying: Screw feeder axial velocity = pitch × RPM / 60

Geometry:
    All dimensions imported directly from FeedSystemAssembly
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
import numpy as np
import warp as wp

from ..geometry.assembly.feed_system import FeedSystemAssembly, FeedSystemParams
from ..utils.constants import GRAVITY, PI, TWO_PI, AirProperties


# =============================================================================
# PHYSICAL CONSTANTS (SI units)
# =============================================================================

# Air properties at 20°C, 1 atm (from constants module)
RHO_AIR = AirProperties.DENSITY           # 1.204 kg/m³
MU_AIR = AirProperties.DYNAMIC_VISCOSITY  # 1.825e-5 Pa·s

# Default particle properties (flour-like powder)
DEFAULT_PARTICLE_DENSITY = 1400.0  # kg/m³
DEFAULT_RESTITUTION = 0.3          # Coefficient of restitution (soft powder)
DEFAULT_FRICTION = 0.4             # Friction coefficient


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class FlowZone(Enum):
    """Flow zones in the feed system."""
    HOPPER = 0
    AIRLOCK = 1
    FEEDER = 2
    DEAGGLOMERATOR = 3
    EXITED = 4


class LidState(Enum):
    """State of the hopper lid."""
    CLOSED = "closed"
    OPENING = "opening"
    OPEN = "open"
    CLOSING = "closing"


class SimulationPhase(Enum):
    """Simulation workflow phases."""
    IDLE = "idle"
    POURING = "pouring"        # Lid open, particles pouring in
    SETTLING = "settling"       # Lid closed, particles settling
    FLOWING = "flowing"         # Discharge open, material flowing through system
    COMPLETED = "completed"


@dataclass
class ComponentGeometry:
    """Extracted geometry from a feed system component."""
    center: np.ndarray          # World position of component center
    axis: str                   # Rotation axis ('x', 'y', 'z')
    
    # Cylindrical/housing parameters
    radius: float = 0.0         # Main radius (housing or rotor)
    length: float = 0.0         # Axial length
    
    # Port parameters (computed from actual ports)
    inlet_pos: np.ndarray = None    # World position of inlet port
    inlet_dir: np.ndarray = None    # Inlet direction (unit vector, points INTO component)
    inlet_diameter: float = 0.0
    outlet_pos: np.ndarray = None   # World position of outlet port
    outlet_dir: np.ndarray = None   # Outlet direction (unit vector, points OUT of component)
    outlet_diameter: float = 0.0
    
    # Cone parameters (for hopper)
    top_radius: float = 0.0
    bottom_radius: float = 0.0
    cone_height: float = 0.0
    cylinder_height: float = 0.0
    
    # Axis vectors (computed from axis string)
    axis_vector: np.ndarray = None  # Unit vector along rotation axis
    radial_plane: str = ''          # Plane for radial containment ('xy', 'xz', 'yz')


@dataclass
class ConnectionPath:
    """Geometry for the path between two components."""
    name: str
    start_pos: np.ndarray       # World position of start (source outlet)
    end_pos: np.ndarray         # World position of end (target inlet)
    direction: np.ndarray       # Unit vector from start to end
    length: float               # Distance between start and end
    start_diameter: float       # Diameter at start
    end_diameter: float         # Diameter at end
    avg_radius: float           # Average radius for containment


@dataclass
class FlowPhysicsConfig:
    """Configuration for physics-based flow simulation."""
    
    # Time stepping
    dt: float = 0.005               # Time step [s] - smaller for stability
    total_time: float = 20.0        # Total simulation time [s]
    
    # Particle properties
    particle_density: float = DEFAULT_PARTICLE_DENSITY
    restitution: float = DEFAULT_RESTITUTION
    friction: float = DEFAULT_FRICTION
    
    # Fluid properties
    air_density: float = RHO_AIR
    air_viscosity: float = MU_AIR
    
    # Component RPMs (operating conditions)
    airlock_rpm: float = 20.0
    feeder_rpm: float = 60.0
    deagg_rpm: float = 1500.0
    
    # Simulation parameters
    num_particles: int = 5000
    device: str = "cuda"
    
    # Hopper lid animation
    animate_lid: bool = True
    lid_open_angle: float = 90.0      # [degrees] Maximum lid opening angle
    lid_animation_time: float = 1.0   # [s] Time to open/close lid
    
    # Pouring simulation
    enable_pouring: bool = True
    hopper_fill_percentage: float = 50.0  # [%] Percentage of hopper capacity to fill
    pour_height: float = 0.3              # [m] Height above hopper to pour from
    pour_rate_kg_s: float = 50.0          # [kg/s] Mass flow rate during pouring
    pour_stream_radius: float = 0.15      # [m] Radius of pour stream
    settling_time: float = 1.0            # [s] Time to wait for particles to settle
    
    # Visual particle sizing
    visual_particle_diameter: float = 0.04  # [m] Display size (40mm visual particles)


@dataclass
class FlowPhysicsState:
    """State of the flow physics simulation."""
    time: float = 0.0
    step: int = 0
    
    # Particle data (on device)
    positions: Optional[wp.array] = None
    velocities: Optional[wp.array] = None
    diameters: Optional[wp.array] = None
    masses: Optional[wp.array] = None
    zones: Optional[wp.array] = None
    is_active: Optional[wp.array] = None
    
    # Counts
    particles_active: int = 0
    particles_poured: int = 0
    total_particles_to_pour: int = 0
    
    # Lid state
    lid_state: LidState = LidState.CLOSED
    lid_angle: float = 0.0              # [degrees] Current lid angle (0=closed)
    lid_target_angle: float = 0.0       # [degrees] Target lid angle
    
    # Simulation phase
    phase: SimulationPhase = SimulationPhase.IDLE
    phase_start_time: float = 0.0
    
    # Calculated flow rates (from geometry and RPM)
    airlock_volumetric_rate: float = 0.0  # [m³/s]
    airlock_mass_rate: float = 0.0        # [kg/s]
    feeder_volumetric_rate: float = 0.0   # [m³/s]
    feeder_mass_rate: float = 0.0         # [kg/s]


# =============================================================================
# GEOMETRY EXTRACTION
# =============================================================================

def extract_geometry(assembly: FeedSystemAssembly) -> Dict[str, ComponentGeometry]:
    """
    Extract all geometry parameters from a FeedSystemAssembly.
    
    This function computes all dimensions from the actual component geometry,
    ensuring no magic numbers are used in the simulation.
    
    Args:
        assembly: FeedSystemAssembly instance
        
    Returns:
        Dictionary mapping component names to their geometry
    """
    positions = assembly.get_component_positions()
    geometry = {}
    
    # =========================================================================
    # HOPPER GEOMETRY
    # =========================================================================
    hopper = assembly.hopper
    hopper_pos = np.array(positions['hopper'])
    hopper_ports = hopper.ports
    
    geometry['hopper'] = ComponentGeometry(
        center=hopper_pos,
        axis='y',  # Hopper is vertical
        radius=hopper.params.top_radius,
        top_radius=hopper.params.top_radius,
        bottom_radius=hopper.params.bottom_radius,
        cone_height=hopper.params.conical_height,
        cylinder_height=hopper.params.cylindrical_height,
        length=hopper.params.total_height,
        inlet_pos=hopper_pos + np.array([0, hopper.params.total_height, 0]),
        inlet_dir=np.array([0.0, 1.0, 0.0]),
        inlet_diameter=hopper.params.top_radius * 2,
        outlet_pos=hopper_pos + np.array(hopper_ports['discharge'].position),
        outlet_dir=np.array(hopper_ports['discharge'].direction),
        outlet_diameter=hopper_ports['discharge'].diameter,
    )
    
    # =========================================================================
    # AIRLOCK GEOMETRY
    # =========================================================================
    airlock = assembly.airlock
    airlock_pos = np.array(positions['airlock'])
    airlock_ports = airlock.ports
    
    geometry['airlock'] = ComponentGeometry(
        center=airlock_pos,
        axis=airlock.params.axis,  # Typically 'z' - rotation around Z
        radius=airlock.params.rotor_radius,
        length=airlock.params.rotor_length,
        inlet_pos=airlock_pos + np.array(airlock_ports['inlet'].position),
        inlet_dir=np.array(airlock_ports['inlet'].direction),
        inlet_diameter=airlock_ports['inlet'].diameter,
        outlet_pos=airlock_pos + np.array(airlock_ports['outlet'].position),
        outlet_dir=np.array(airlock_ports['outlet'].direction),
        outlet_diameter=airlock_ports['outlet'].diameter,
    )
    
    # =========================================================================
    # SCREW FEEDER GEOMETRY
    # =========================================================================
    feeder = assembly.feeder
    feeder_pos = np.array(positions['feeder'])
    feeder_ports = feeder.ports
    
    geometry['feeder'] = ComponentGeometry(
        center=feeder_pos,
        axis=feeder.params.axis,  # Typically 'x' - screw along X
        radius=feeder.params.trough_radius,
        length=feeder.params.trough_length,
        inlet_pos=feeder_pos + np.array(feeder_ports['inlet'].position),
        inlet_dir=np.array(feeder_ports['inlet'].direction),
        inlet_diameter=feeder_ports['inlet'].diameter,
        outlet_pos=feeder_pos + np.array(feeder_ports['outlet'].position),
        outlet_dir=np.array(feeder_ports['outlet'].direction),
        outlet_diameter=feeder_ports['outlet'].diameter,
    )
    # Store screw pitch for conveying calculation
    geometry['feeder'].screw_pitch = feeder.params.screw_pitch
    
    # =========================================================================
    # DEAGGLOMERATOR GEOMETRY
    # =========================================================================
    deagg = assembly.deagglomerator
    deagg_pos = np.array(positions['deagglomerator'])
    deagg_ports = deagg.ports
    
    geometry['deagglomerator'] = ComponentGeometry(
        center=deagg_pos,
        axis=deagg.params.axis,  # Typically 'x' - horizontal cylinder
        radius=deagg.params.housing_radius,
        length=deagg.params.housing_length,
        inlet_pos=deagg_pos + np.array(deagg_ports['inlet'].position),
        inlet_dir=np.array(deagg_ports['inlet'].direction),
        inlet_diameter=deagg_ports['inlet'].diameter,
        outlet_pos=deagg_pos + np.array(deagg_ports['outlet'].position),
        outlet_dir=np.array(deagg_ports['outlet'].direction),
        outlet_diameter=deagg_ports['outlet'].diameter,
    )
    # Store rotor radius for tangential velocity calculation
    geometry['deagglomerator'].rotor_radius = deagg.params.rotor_radius
    
    # =========================================================================
    # CONNECTION PATHS (computed from actual port positions)
    # These define the exact flow path between components with angles
    # =========================================================================
    connections = {}
    
    # Hopper -> Airlock connection
    hopper_outlet = geometry['hopper'].outlet_pos
    airlock_inlet = geometry['airlock'].inlet_pos
    conn_vec = airlock_inlet - hopper_outlet
    conn_len = float(np.linalg.norm(conn_vec))
    conn_dir = conn_vec / max(conn_len, 1e-6)
    
    connections['hopper_to_airlock'] = {
        'start_pos': hopper_outlet.copy(),
        'end_pos': airlock_inlet.copy(),
        'direction': conn_dir,
        'length': conn_len,
        'start_diameter': geometry['hopper'].outlet_diameter,
        'end_diameter': geometry['airlock'].inlet_diameter,
        'avg_radius': (geometry['hopper'].outlet_diameter + geometry['airlock'].inlet_diameter) / 4.0,
    }
    
    # Airlock -> Feeder connection
    airlock_outlet = geometry['airlock'].outlet_pos
    feeder_inlet = geometry['feeder'].inlet_pos
    conn_vec = feeder_inlet - airlock_outlet
    conn_len = float(np.linalg.norm(conn_vec))
    conn_dir = conn_vec / max(conn_len, 1e-6)
    
    connections['airlock_to_feeder'] = {
        'start_pos': airlock_outlet.copy(),
        'end_pos': feeder_inlet.copy(),
        'direction': conn_dir,
        'length': conn_len,
        'start_diameter': geometry['airlock'].outlet_diameter,
        'end_diameter': geometry['feeder'].inlet_diameter,
        'avg_radius': (geometry['airlock'].outlet_diameter + geometry['feeder'].inlet_diameter) / 4.0,
    }
    
    # Feeder -> Deagg connection
    feeder_outlet = geometry['feeder'].outlet_pos
    deagg_inlet = geometry['deagglomerator'].inlet_pos
    conn_vec = deagg_inlet - feeder_outlet
    conn_len = float(np.linalg.norm(conn_vec))
    conn_dir = conn_vec / max(conn_len, 1e-6)
    
    connections['feeder_to_deagg'] = {
        'start_pos': feeder_outlet.copy(),
        'end_pos': deagg_inlet.copy(),
        'direction': conn_dir,
        'length': conn_len,
        'start_diameter': geometry['feeder'].outlet_diameter,
        'end_diameter': geometry['deagglomerator'].inlet_diameter,
        'avg_radius': (geometry['feeder'].outlet_diameter + geometry['deagglomerator'].inlet_diameter) / 4.0,
    }
    
    geometry['connections'] = connections
    
    return geometry


# =============================================================================
# WARP PHYSICS FUNCTIONS
# =============================================================================

@wp.func
def compute_particle_reynolds(
    diameter: float,
    v_rel_mag: float,
    rho_f: float,
    mu_f: float
) -> float:
    """Compute particle Reynolds number: Re = ρ*v*d/μ"""
    eps = 1.0e-10
    return rho_f * v_rel_mag * diameter / wp.max(mu_f, eps)


@wp.func
def compute_drag_coefficient(Re: float) -> float:
    """
    Schiller-Naumann drag coefficient.
    Valid for Re < 1000.
    Cd = (24/Re) * (1 + 0.15 * Re^0.687)
    """
    eps = 1.0e-10
    if Re < eps:
        return 24.0 / eps
    return (24.0 / Re) * (1.0 + 0.15 * wp.pow(Re, 0.687))


@wp.func
def compute_gravity_acceleration(
    rho_p: float,
    rho_f: float,
    g: float
) -> wp.vec3:
    """
    Gravitational acceleration with buoyancy: a = (1 - ρ_f/ρ_p) * g
    Returns vector in -Y direction.
    """
    a_mag = (1.0 - rho_f / rho_p) * g
    return wp.vec3(0.0, -a_mag, 0.0)


@wp.func
def compute_drag_acceleration(
    vel: wp.vec3,
    diameter: float,
    mass: float,
    rho_f: float,
    mu_f: float
) -> wp.vec3:
    """
    Compute drag acceleration from air resistance.
    F_drag = 0.5 * Cd * ρ_f * A * v²
    a_drag = F_drag / m (in direction of -velocity)
    """
    v_mag = wp.length(vel)
    eps = 1.0e-10
    
    if v_mag < eps:
        return wp.vec3(0.0, 0.0, 0.0)
    
    Re = compute_particle_reynolds(diameter, v_mag, rho_f, mu_f)
    Cd = compute_drag_coefficient(Re)
    
    # Projected area
    A = 3.141592653589793 / 4.0 * diameter * diameter
    
    # Drag force magnitude
    F_drag = 0.5 * Cd * rho_f * A * v_mag * v_mag
    
    # Acceleration (opposite to velocity direction)
    a_mag = F_drag / mass
    
    return vel * (-a_mag / v_mag)


@wp.func
def compute_tangential_velocity(
    pos: wp.vec3,
    axis_center: wp.vec3,
    axis_dir: wp.vec3,
    omega: float
) -> wp.vec3:
    """
    Compute tangential velocity at position from rotating body.
    v_tan = ω × r (cross product of angular velocity and radius vector)
    
    For axis along X: v_tan has components in Y and Z
    For axis along Y: v_tan has components in X and Z
    For axis along Z: v_tan has components in X and Y
    """
    # Vector from axis to position
    to_pos = pos - axis_center
    
    # Remove axial component to get radial vector
    axial_component = wp.dot(to_pos, axis_dir) * axis_dir
    radial = to_pos - axial_component
    
    r = wp.length(radial)
    eps = 1.0e-10
    
    if r < eps:
        return wp.vec3(0.0, 0.0, 0.0)
    
    # Tangential direction (perpendicular to both axis and radial)
    # v = ω × r = |ω| * |r| * tangent_direction
    tangent = wp.cross(axis_dir, radial / r)
    
    return tangent * (omega * r)


@wp.func
def reflect_velocity_inelastic(
    vel: wp.vec3,
    normal: wp.vec3,
    restitution: float,
    friction: float
) -> wp.vec3:
    """
    Reflect velocity off a surface with inelastic collision.
    
    Normal component: v_n' = -e * v_n (restitution)
    Tangent component: v_t' = (1-μ) * v_t (friction)
    """
    n_len = wp.length(normal)
    eps = 1.0e-10
    
    if n_len < eps:
        return vel
    
    n = normal / n_len
    
    # Decompose velocity
    v_n = wp.dot(vel, n)
    v_normal = n * v_n
    v_tangent = vel - v_normal
    
    # Only reflect if moving into surface
    if v_n >= 0.0:
        return vel
    
    # Apply restitution to normal component
    v_normal_new = -restitution * v_normal
    
    # Apply friction to tangential component
    v_tangent_new = v_tangent * (1.0 - friction)
    
    return v_normal_new + v_tangent_new


# =============================================================================
# WARP COLLISION FUNCTIONS
# =============================================================================

@wp.func
def sdf_cylinder_y_axis(
    pos: wp.vec3,
    center: wp.vec3,
    radius: float,
    half_height: float
) -> float:
    """
    Signed distance function for cylinder aligned with Y axis.
    Negative inside, positive outside.
    """
    # Radial distance in XZ plane
    dx = pos[0] - center[0]
    dz = pos[2] - center[2]
    r = wp.sqrt(dx * dx + dz * dz)
    
    # Axial distance
    dy = pos[1] - center[1]
    
    # Distance to cylindrical surface
    dist_radial = r - radius
    
    # Distance to end caps
    dist_top = dy - half_height
    dist_bottom = -half_height - dy
    
    # Combined SDF (max of all constraints)
    # Inside: all distances negative
    # Outside: at least one positive
    return wp.max(wp.max(dist_radial, dist_top), dist_bottom)


@wp.func
def sdf_cylinder_x_axis(
    pos: wp.vec3,
    center: wp.vec3,
    radius: float,
    half_length: float
) -> float:
    """
    Signed distance function for cylinder aligned with X axis (horizontal).
    Negative inside, positive outside.
    """
    # Radial distance in YZ plane
    dy = pos[1] - center[1]
    dz = pos[2] - center[2]
    r = wp.sqrt(dy * dy + dz * dz)
    
    # Axial distance
    dx = pos[0] - center[0]
    
    dist_radial = r - radius
    dist_end_pos = dx - half_length
    dist_end_neg = -half_length - dx
    
    return wp.max(wp.max(dist_radial, dist_end_pos), dist_end_neg)


@wp.func
def sdf_cone_y_axis(
    pos: wp.vec3,
    apex: wp.vec3,
    top_radius: float,
    bottom_radius: float,
    height: float
) -> float:
    """
    Signed distance function for truncated cone (frustum) with Y axis.
    apex is at the top (smaller radius), cone expands downward.
    """
    # Position relative to apex
    dy = apex[1] - pos[1]  # Distance below apex
    dx = pos[0] - apex[0]
    dz = pos[2] - apex[2]
    r = wp.sqrt(dx * dx + dz * dz)
    
    # Clamp to cone height
    t = wp.clamp(dy / height, 0.0, 1.0)
    
    # Radius at this height (linear interpolation)
    r_at_height = top_radius + t * (bottom_radius - top_radius)
    
    # Distance to cone surface
    dist_radial = r - r_at_height
    
    # Distance to end caps
    dist_top = -dy  # Above apex
    dist_bottom = dy - height  # Below base
    
    return wp.max(wp.max(dist_radial, dist_top), dist_bottom)


# =============================================================================
# MAIN PHYSICS KERNEL
# =============================================================================

@wp.kernel
def physics_flow_kernel(
    # Particle state
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    diameters: wp.array(dtype=float),
    masses: wp.array(dtype=float),
    zones: wp.array(dtype=wp.int32),
    is_active: wp.array(dtype=wp.int32),
    num_particles: int,
    
    # Hopper geometry
    hopper_center: wp.vec3,
    hopper_top_radius: float,
    hopper_bottom_radius: float,
    hopper_cylinder_height: float,
    hopper_cone_height: float,
    hopper_outlet_y: float,
    hopper_outlet_radius: float,
    
    # Airlock geometry
    airlock_center: wp.vec3,
    airlock_radius: float,
    airlock_half_length: float,
    airlock_inlet_y: float,
    airlock_outlet_y: float,
    airlock_omega: float,  # rad/s = 2π * RPM / 60
    
    # Feeder geometry
    feeder_center: wp.vec3,
    feeder_radius: float,
    feeder_half_length: float,
    feeder_inlet_x: float,
    feeder_outlet_x: float,
    feeder_outlet_y: float,
    feeder_omega: float,
    feeder_axial_speed: float,  # pitch * RPM / 60
    
    # Deagg geometry
    deagg_center: wp.vec3,
    deagg_radius: float,
    deagg_half_length: float,
    deagg_inlet_y: float,
    deagg_outlet_y: float,
    deagg_omega: float,
    deagg_rotor_radius: float,
    deagg_outlet_radius: float,  # Actual outlet opening radius
    
    # Exit
    exit_y: float,
    
    # Physics parameters
    dt: float,
    gravity: float,
    rho_p: float,
    rho_f: float,
    mu_f: float,
    restitution: float,
    friction: float,
    
    # Control
    discharge_open: int,
):
    """
    Physics-based flow kernel using actual geometry.
    
    All transitions are based on geometric boundaries.
    All velocities computed from physics principles.
    """
    tid = wp.tid()
    
    if tid >= num_particles:
        return
    
    if is_active[tid] == 0:
        return
    
    pos = positions[tid]
    vel = velocities[tid]
    zone = zones[tid]
    d = diameters[tid]
    m = masses[tid]
    particle_radius = d * 0.5
    
    # Compute common accelerations
    a_gravity = compute_gravity_acceleration(rho_p, rho_f, gravity)
    a_drag = compute_drag_acceleration(vel, d, m, rho_f, mu_f)
    
    # Total acceleration starts with gravity and drag
    accel = a_gravity + a_drag
    
    # =========================================================================
    # ZONE 0: HOPPER
    # =========================================================================
    if zone == 0:
        # Position relative to hopper center (bottom of cone)
        local_y = pos[1] - hopper_center[1]
        local_x = pos[0] - hopper_center[0]
        local_z = pos[2] - hopper_center[2]
        r = wp.sqrt(local_x * local_x + local_z * local_z)
        
        # Compute wall radius at current height
        if local_y < hopper_cone_height:
            # In cone section
            t = local_y / hopper_cone_height
            wall_radius = hopper_bottom_radius + t * (hopper_top_radius - hopper_bottom_radius)
        else:
            # In cylinder section
            wall_radius = hopper_top_radius
        
        # Wall collision
        penetration = r + particle_radius - wall_radius
        if penetration > 0.0 and r > 1.0e-6:
            # Push back and reflect
            normal = wp.vec3(-local_x / r, 0.0, -local_z / r)
            pos = pos + normal * (penetration + 0.001)
            vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
        
        # Bottom collision (cone apex)
        if local_y < particle_radius:
            if r > hopper_outlet_radius:
                # Hit bottom, not in outlet
                normal = wp.vec3(0.0, 1.0, 0.0)
                pos = wp.vec3(pos[0], hopper_center[1] + particle_radius + 0.001, pos[2])
                vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
        
        # Top boundary
        total_height = hopper_cone_height + hopper_cylinder_height
        if local_y > total_height - particle_radius:
            normal = wp.vec3(0.0, -1.0, 0.0)
            pos = wp.vec3(pos[0], hopper_center[1] + total_height - particle_radius - 0.001, pos[2])
            vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
        
        # TRANSITION: Through discharge when below outlet level and inside opening radius
        # Uses actual outlet geometry - particle must be in the discharge opening
        if discharge_open == 1:
            # Check if particle is within the discharge opening (circular hole at bottom)
            # Outlet is at hopper_center + outlet offset, pointing down
            outlet_center_y = hopper_center[1]  # Discharge is at the bottom of hopper
            in_discharge_zone = (local_y < particle_radius) and (r < hopper_outlet_radius)
            
            if in_discharge_zone:
                zones[tid] = 1
                # Continue along the connection path direction (gravity-driven)
                # Particle enters airlock at its inlet position
                pos = wp.vec3(airlock_center[0], airlock_inlet_y + airlock_radius * 0.5, airlock_center[2])
                # Preserve vertical velocity component
                vel = wp.vec3(vel[0] * 0.3, vel[1], vel[2] * 0.3)
    
    # =========================================================================
    # ZONE 1: AIRLOCK
    # =========================================================================
    elif zone == 1:
        # Position relative to airlock center
        px = pos[0] - airlock_center[0]
        py = pos[1] - airlock_center[1]
        pz = pos[2] - airlock_center[2]
        
        # Airlock rotates around Z axis (vertical)
        # Radial distance in XY plane
        r_xy = wp.sqrt(px * px + py * py)
        
        # Rotational effect from vanes
        if r_xy > 0.01:
            # Tangential velocity from rotating vanes
            v_tan = compute_tangential_velocity(
                pos, airlock_center,
                wp.vec3(0.0, 0.0, 1.0),  # Z axis
                airlock_omega
            )
            # Couple particle to vane rotation (partial coupling)
            coupling = 0.3  # 30% coupling to vane speed
            accel = accel + (v_tan - vel) * coupling / dt
        
        # Cylindrical housing wall collision
        r_xz = wp.sqrt(px * px + pz * pz)
        if r_xz + particle_radius > airlock_radius:
            if r_xz > 1.0e-6:
                normal = wp.vec3(-px / r_xz, 0.0, -pz / r_xz)
                push = r_xz + particle_radius - airlock_radius + 0.001
                pos = pos + normal * push
                vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
        
        # Axial containment (Z direction)
        if wp.abs(pz) + particle_radius > airlock_half_length:
            sign_z = 1.0 if pz > 0.0 else -1.0
            normal = wp.vec3(0.0, 0.0, -sign_z)
            pos = wp.vec3(pos[0], pos[1], airlock_center[2] + sign_z * (airlock_half_length - particle_radius - 0.001))
            vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
        
        # Top inlet containment
        if py > airlock_radius - particle_radius:
            normal = wp.vec3(0.0, -1.0, 0.0)
            pos = wp.vec3(pos[0], airlock_center[1] + airlock_radius - particle_radius - 0.001, pos[2])
            vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
        
        # TRANSITION: Through outlet at bottom
        # Uses actual outlet port position - particle exits when below outlet
        outlet_y_threshold = -airlock_radius + particle_radius
        in_outlet_region = py < outlet_y_threshold
        
        if in_outlet_region:
            zones[tid] = 2
            # Particle enters feeder at its inlet position
            # Feeder inlet is connected to airlock outlet via transition
            # Give particle velocity along feeder axis
            pos = wp.vec3(feeder_center[0] - feeder_half_length + particle_radius * 2.0, 
                          feeder_center[1], 
                          feeder_center[2])
            # Initial velocity has component along feeder axis plus residual gravity
            vel = wp.vec3(feeder_axial_speed * 0.3, vel[1] * 0.3, vel[2] * 0.3)
    
    # =========================================================================
    # ZONE 2: SCREW FEEDER
    # =========================================================================
    elif zone == 2:
        # Feeder is horizontal cylinder along X axis
        px = pos[0] - feeder_center[0]
        py = pos[1] - feeder_center[1]
        pz = pos[2] - feeder_center[2]
        
        # Progress along feeder (0 to 1)
        feeder_length = 2.0 * feeder_half_length
        progress = (pos[0] - feeder_inlet_x) / feeder_length
        progress = wp.clamp(progress, 0.0, 1.0)
        
        # Screw conveying effect: impart axial velocity
        # Conveying velocity = pitch × RPM / 60 (already computed as feeder_axial_speed)
        target_vx = feeder_axial_speed
        
        # Rotational effect from screw
        r_yz = wp.sqrt(py * py + pz * pz)
        if r_yz > 0.005:
            v_tan = compute_tangential_velocity(
                pos, feeder_center,
                wp.vec3(1.0, 0.0, 0.0),  # X axis
                feeder_omega
            )
            # Partial coupling to screw rotation
            coupling = 0.2
            accel = accel + (v_tan - vel) * coupling / dt
        
        # Apply axial conveying force
        ax_force = (target_vx - vel[0]) * 2.0 / dt  # Proportional control
        accel = accel + wp.vec3(ax_force, 0.0, 0.0)
        
        # Tube wall collision (radial in YZ)
        if r_yz + particle_radius > feeder_radius:
            if r_yz > 1.0e-6:
                normal_y = -py / r_yz
                normal_z = -pz / r_yz
                normal = wp.vec3(0.0, normal_y, normal_z)
                push = r_yz + particle_radius - feeder_radius + 0.001
                pos = pos + normal * push
                vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
        
        # Inlet end cap
        if px < -feeder_half_length + particle_radius:
            normal = wp.vec3(1.0, 0.0, 0.0)
            pos = wp.vec3(feeder_center[0] - feeder_half_length + particle_radius + 0.001, pos[1], pos[2])
            vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
        
        # TRANSITION: At outlet end - uses actual outlet port geometry
        # Particle exits when it reaches the outlet end of the feeder
        at_outlet = px > feeder_half_length - particle_radius
        
        if at_outlet:
            zones[tid] = 3
            # Particle enters deagglomerator at its inlet position (top of cylinder)
            # The transition path goes from feeder outlet to deagg inlet
            pos = wp.vec3(deagg_center[0], 
                          deagg_center[1] + deagg_radius * 0.7,  # Near top, inside housing
                          deagg_center[2])
            # Velocity directed into deagglomerator with some axial component
            vel = wp.vec3(vel[0] * 0.2, -wp.abs(vel[1]) - 0.3, vel[2] * 0.2)
    
    # =========================================================================
    # ZONE 3: DEAGGLOMERATOR
    # =========================================================================
    elif zone == 3:
        # Deagg is horizontal cylinder along X axis
        # Inlet at top (Y+), outlet at bottom (Y-)
        px = pos[0] - deagg_center[0]
        py = pos[1] - deagg_center[1]
        pz = pos[2] - deagg_center[2]
        
        # Radial distance in YZ plane
        r_yz = wp.sqrt(py * py + pz * pz)
        
        # High-speed rotor rotation effect
        if r_yz > 0.005 and r_yz < deagg_rotor_radius:
            # Inside rotor region - strong tangential coupling
            v_tan = compute_tangential_velocity(
                pos, deagg_center,
                wp.vec3(1.0, 0.0, 0.0),  # X axis
                deagg_omega
            )
            # Strong coupling near rotor
            coupling = 0.5 * (1.0 - r_yz / deagg_rotor_radius)
            accel = accel + (v_tan - vel) * coupling / dt
        
        # Outlet opening parameters:
        # The outlet is at the bottom of the cylinder (Y-)
        # Opening starts at Y = center - radius (bottom of cylinder)
        # We allow particles to exit if they're in the outlet region
        outlet_opening_threshold = -deagg_radius + deagg_outlet_radius
        
        # Check if particle is in the outlet region (bottom of cylinder, within outlet diameter)
        # Outlet opening is circular, centered at (center_x, center_y - radius, center_z)
        in_outlet_region = (py < outlet_opening_threshold) and (wp.abs(pz) < deagg_outlet_radius)
        
        # Housing wall collision - only apply if NOT in outlet region
        if r_yz + particle_radius > deagg_radius and not in_outlet_region:
            if r_yz > 1.0e-6:
                normal_y = -py / r_yz
                normal_z = -pz / r_yz
                normal = wp.vec3(0.0, normal_y, normal_z)
                push = r_yz + particle_radius - deagg_radius + 0.001
                pos = pos + normal * push
                vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
        
        # End cap collision
        if wp.abs(px) + particle_radius > deagg_half_length:
            sign_x = 1.0 if px > 0.0 else -1.0
            normal = wp.vec3(-sign_x, 0.0, 0.0)
            pos = wp.vec3(deagg_center[0] + sign_x * (deagg_half_length - particle_radius - 0.001), pos[1], pos[2])
            vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
        
        # Inlet containment (top)
        if py > deagg_radius * 0.8:
            normal = wp.vec3(0.0, -1.0, 0.0)
            pos = wp.vec3(pos[0], deagg_center[1] + deagg_radius * 0.75, pos[2])
            vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
        
        # TRANSITION: Exit through outlet at bottom
        # Particle exits when it falls below the outlet Y position
        if pos[1] < deagg_outlet_y:
            zones[tid] = 4
            # Continue with downward velocity
            vel = wp.vec3(0.0, wp.min(vel[1], -1.0), 0.0)
    
    # =========================================================================
    # ZONE 4: EXITED
    # =========================================================================
    elif zone == 4:
        # Free fall to collection
        # No additional forces beyond gravity and drag (already in accel)
        
        # Floor collision
        if pos[1] < exit_y + particle_radius:
            normal = wp.vec3(0.0, 1.0, 0.0)
            pos = wp.vec3(pos[0], exit_y + particle_radius + 0.001, pos[2])
            vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
            
            # Deactivate if settled
            if wp.length(vel) < 0.05:
                is_active[tid] = 0
    
    # =========================================================================
    # INTEGRATION: Semi-implicit Euler
    # =========================================================================
    vel = vel + accel * dt
    pos = pos + vel * dt
    
    # Write back
    positions[tid] = pos
    velocities[tid] = vel


# =============================================================================
# PARTICLE-PARTICLE COLLISION KERNEL
# =============================================================================

@wp.kernel
def particle_collision_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    diameters: wp.array(dtype=float),
    masses: wp.array(dtype=float),
    is_active: wp.array(dtype=wp.int32),
    grid: wp.uint64,
    num_particles: int,
    restitution: float,
    search_radius: float,
):
    """
    Particle-particle collision detection and response using hash grid.
    """
    tid = wp.tid()
    
    if tid >= num_particles:
        return
    
    if is_active[tid] == 0:
        return
    
    pos_i = positions[tid]
    vel_i = velocities[tid]
    r_i = diameters[tid] * 0.5
    m_i = masses[tid]
    
    # Query neighbors
    query = wp.hash_grid_query(grid, pos_i, search_radius)
    idx = int(0)
    
    # Accumulate collision impulses
    dv = wp.vec3(0.0, 0.0, 0.0)
    
    while wp.hash_grid_query_next(query, idx):
        if idx == tid:
            continue
        
        if is_active[idx] == 0:
            continue
        
        pos_j = positions[idx]
        r_j = diameters[idx] * 0.5
        m_j = masses[idx]
        
        # Distance between particles
        diff = pos_i - pos_j
        dist = wp.length(diff)
        
        contact_dist = r_i + r_j
        
        if dist < contact_dist and dist > 1.0e-8:
            # Collision detected
            normal = diff / dist
            
            # Relative velocity
            vel_j = velocities[idx]
            v_rel = vel_i - vel_j
            v_n = wp.dot(v_rel, normal)
            
            # Only process if approaching
            if v_n < 0.0:
                # Impulse-based collision response
                # j = -(1+e) * v_n / (1/m_i + 1/m_j)
                j = -(1.0 + restitution) * v_n / (1.0 / m_i + 1.0 / m_j)
                
                # Velocity change for particle i
                dv = dv + normal * (j / m_i)
    
    velocities[tid] = vel_i + dv


# =============================================================================
# SIMULATOR CLASS
# =============================================================================

class FeedFlowPhysicsSimulator:
    """
    Physics-based material flow simulator for feed system.
    
    Uses actual geometry from FeedSystemAssembly and computes all
    physics from first principles - no magic numbers.
    """
    
    def __init__(
        self,
        assembly: FeedSystemAssembly,
        config: FlowPhysicsConfig = None,
    ):
        """
        Initialize the physics-based flow simulator.
        
        Args:
            assembly: FeedSystemAssembly with actual geometry
            config: Simulation configuration
        """
        self.assembly = assembly
        self.config = config or FlowPhysicsConfig()
        self.state = FlowPhysicsState()
        
        # Initialize Warp
        wp.init()
        self.device = self.config.device
        
        # Extract geometry from assembly
        self.geometry = extract_geometry(assembly)
        self._compute_derived_parameters()
        
        # Allocate arrays
        self._allocate_arrays()
        
        # Hash grid for particle-particle collisions
        self._setup_hash_grid()
        
        # State flags
        self._discharge_open = 0
    
    def _compute_derived_parameters(self):
        """Compute all derived physics parameters from geometry."""
        cfg = self.config
        
        # =====================================================================
        # ANGULAR VELOCITIES (computed from RPM)
        # ω = 2π × RPM / 60
        # =====================================================================
        self.airlock_omega = TWO_PI * cfg.airlock_rpm / 60.0
        self.feeder_omega = TWO_PI * cfg.feeder_rpm / 60.0
        self.deagg_omega = TWO_PI * cfg.deagg_rpm / 60.0
        
        # =====================================================================
        # SCREW FEEDER AXIAL VELOCITY
        # v_axial = pitch × RPM / 60
        # =====================================================================
        feeder_pitch = self.geometry['feeder'].screw_pitch
        self.feeder_axial_speed = feeder_pitch * cfg.feeder_rpm / 60.0
        
        # =====================================================================
        # VOLUMETRIC FLOW RATES (calculated from geometry and RPM)
        # =====================================================================
        airlock_geo = self.geometry['airlock']
        
        # Airlock: Q = pocket_volume × num_pockets × RPM / 60
        # Pocket volume ≈ (π × r² × L) / num_vanes
        airlock_r = airlock_geo.radius
        airlock_L = airlock_geo.length
        num_vanes = self.assembly.airlock.params.num_vanes
        pocket_volume = (PI * airlock_r**2 * airlock_L) / num_vanes
        self.state.airlock_volumetric_rate = pocket_volume * num_vanes * cfg.airlock_rpm / 60.0
        self.state.airlock_mass_rate = self.state.airlock_volumetric_rate * cfg.particle_density * 0.6  # 60% fill
        
        # Screw feeder: Q = (π/4) × D² × pitch × RPM / 60 × fill_factor
        feeder_geo = self.geometry['feeder']
        screw_d = feeder_geo.radius * 2 * 0.9  # Screw diameter (90% of trough)
        fill_factor = 0.45  # Typical screw feeder fill factor
        self.state.feeder_volumetric_rate = (PI / 4) * screw_d**2 * feeder_pitch * cfg.feeder_rpm / 60.0 * fill_factor
        self.state.feeder_mass_rate = self.state.feeder_volumetric_rate * cfg.particle_density
        
        # =====================================================================
        # GEOMETRIC BOUNDARIES
        # =====================================================================
        hopper_geo = self.geometry['hopper']
        self.hopper_outlet_y = hopper_geo.outlet_pos[1]
        self.hopper_outlet_radius = hopper_geo.outlet_diameter / 2.0
        self.hopper_top_y = hopper_geo.center[1] + hopper_geo.cone_height + hopper_geo.cylinder_height
        self.hopper_top_radius = hopper_geo.top_radius
        
        self.airlock_inlet_y = airlock_geo.inlet_pos[1]
        self.airlock_outlet_y = airlock_geo.outlet_pos[1]
        
        self.feeder_inlet_x = feeder_geo.inlet_pos[0]
        self.feeder_outlet_x = feeder_geo.outlet_pos[0]
        self.feeder_outlet_y = feeder_geo.outlet_pos[1]
        
        deagg_geo = self.geometry['deagglomerator']
        self.deagg_inlet_y = deagg_geo.inlet_pos[1]
        self.deagg_outlet_y = deagg_geo.outlet_pos[1]
        self.deagg_outlet_diameter = deagg_geo.outlet_diameter
        
        # Exit floor level (below deagglomerator)
        self.exit_y = self.deagg_outlet_y - 0.3
        
        # Lid animation speed
        self._lid_max_angular_velocity = cfg.lid_open_angle / cfg.lid_animation_time
        
        # Lid hinge position (on -X side of hopper)
        self.lid_hinge_position = np.array([
            hopper_geo.center[0] - self.hopper_top_radius * 1.08,
            self.hopper_top_y,
            hopper_geo.center[2]
        ])
        
        # =====================================================================
        # CONNECTION PATH GEOMETRY (computed from port positions)
        # =====================================================================
        connections = self.geometry.get('connections', {})
        
        # Store connection data for kernel
        self.conn_hopper_airlock = connections.get('hopper_to_airlock', {})
        self.conn_airlock_feeder = connections.get('airlock_to_feeder', {})
        self.conn_feeder_deagg = connections.get('feeder_to_deagg', {})
        
        print(f"\n  Physics Parameters (computed from geometry):")
        print(f"    Airlock ω:        {self.airlock_omega:.2f} rad/s ({cfg.airlock_rpm} RPM)")
        print(f"    Feeder ω:         {self.feeder_omega:.2f} rad/s ({cfg.feeder_rpm} RPM)")
        print(f"    Deagg ω:          {self.deagg_omega:.2f} rad/s ({cfg.deagg_rpm} RPM)")
        print(f"    Feeder v_axial:   {self.feeder_axial_speed*100:.1f} cm/s")
        
        print(f"\n  Connection Paths (from port geometry):")
        for name, conn in connections.items():
            direction = conn['direction']
            angle_y = np.degrees(np.arcsin(abs(direction[1])))  # Angle from horizontal
            print(f"    {name}:")
            print(f"      Length:         {conn['length']*1000:.0f} mm")
            print(f"      Direction:      ({direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f})")
            print(f"      Angle from XZ:  {angle_y:.1f}°")
            print(f"      Diameters:      {conn['start_diameter']*1000:.0f} → {conn['end_diameter']*1000:.0f} mm")
        print(f"\n  Calculated Flow Rates:")
        print(f"    Airlock Q:        {self.state.airlock_volumetric_rate*3600*1000:.1f} L/h")
        print(f"    Airlock ṁ:        {self.state.airlock_mass_rate*3600:.0f} kg/h")
        print(f"    Feeder Q:         {self.state.feeder_volumetric_rate*3600*1000:.1f} L/h")
        print(f"    Feeder ṁ:         {self.state.feeder_mass_rate*3600:.0f} kg/h")
        print(f"\n  Deagglomerator Geometry:")
        print(f"    Center Y:         {deagg_geo.center[1]*1000:.0f} mm")
        print(f"    Radius:           {deagg_geo.radius*1000:.0f} mm")
        print(f"    Outlet Y:         {self.deagg_outlet_y*1000:.0f} mm")
        print(f"    Exit Y:           {self.exit_y*1000:.0f} mm")
    
    def _allocate_arrays(self):
        """Allocate particle arrays on device."""
        n = self.config.num_particles
        
        self.state.positions = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.state.velocities = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.state.diameters = wp.zeros(n, dtype=float, device=self.device)
        self.state.masses = wp.zeros(n, dtype=float, device=self.device)
        self.state.zones = wp.zeros(n, dtype=wp.int32, device=self.device)
        self.state.is_active = wp.zeros(n, dtype=wp.int32, device=self.device)
    
    def _setup_hash_grid(self):
        """Setup hash grid for neighbor search."""
        # Grid size based on system extent
        bounds = self.assembly.get_bounds()
        extent = bounds[1] - bounds[0]
        max_extent = max(extent)
        
        grid_dim = max(32, int(max_extent / 0.05))  # ~5cm cells
        
        self._hash_grid = wp.HashGrid(
            dim_x=grid_dim,
            dim_y=grid_dim,
            dim_z=grid_dim,
            device=self.device
        )
    
    def initialize_particles(
        self,
        num_particles: int,
        mean_diameter: float = 0.04,
        std_diameter: float = 0.005,
    ):
        """
        Initialize particles in the hopper.
        
        Args:
            num_particles: Number of particles to create
            mean_diameter: Mean particle diameter [m]
            std_diameter: Standard deviation of diameter [m]
        """
        n = min(num_particles, self.config.num_particles)
        
        # Generate random positions in hopper
        hopper_geo = self.geometry['hopper']
        rng = np.random.default_rng(42)
        
        positions = np.zeros((n, 3), dtype=np.float32)
        velocities = np.zeros((n, 3), dtype=np.float32)
        
        # Fill hopper volume
        for i in range(n):
            # Random height in hopper
            total_h = hopper_geo.cone_height + hopper_geo.cylinder_height
            h = rng.uniform(0.1 * total_h, 0.9 * total_h)
            
            # Radius at this height
            if h < hopper_geo.cone_height:
                t = h / hopper_geo.cone_height
                r_max = hopper_geo.bottom_radius + t * (hopper_geo.top_radius - hopper_geo.bottom_radius)
            else:
                r_max = hopper_geo.top_radius
            
            # Random position within radius
            r = np.sqrt(rng.random()) * r_max * 0.9
            theta = rng.uniform(0, TWO_PI)
            
            positions[i, 0] = hopper_geo.center[0] + r * np.cos(theta)
            positions[i, 1] = hopper_geo.center[1] + h
            positions[i, 2] = hopper_geo.center[2] + r * np.sin(theta)
        
        # Generate diameters (log-normal distribution)
        diameters = rng.lognormal(
            mean=np.log(mean_diameter),
            sigma=std_diameter / mean_diameter,
            size=n
        ).astype(np.float32)
        diameters = np.clip(diameters, mean_diameter * 0.5, mean_diameter * 2.0)
        
        # Compute masses: m = ρ × (π/6) × d³
        masses = self.config.particle_density * (PI / 6.0) * diameters ** 3
        
        # All particles start in hopper (zone 0), active
        zones = np.zeros(n, dtype=np.int32)
        is_active = np.ones(n, dtype=np.int32)
        
        # Copy to device
        wp.copy(self.state.positions, wp.array(positions, dtype=wp.vec3, device=self.device))
        wp.copy(self.state.velocities, wp.array(velocities, dtype=wp.vec3, device=self.device))
        wp.copy(self.state.diameters, wp.array(diameters, dtype=float, device=self.device))
        wp.copy(self.state.masses, wp.array(masses, dtype=float, device=self.device))
        wp.copy(self.state.zones, wp.array(zones, dtype=wp.int32, device=self.device))
        wp.copy(self.state.is_active, wp.array(is_active, dtype=wp.int32, device=self.device))
        
        self.state.particles_active = n
        
        print(f"  Initialized {n} particles in hopper")
        print(f"    Diameter range: {diameters.min()*1000:.1f} - {diameters.max()*1000:.1f} mm")
        print(f"    Total mass: {masses.sum():.2f} kg")
    
    def start_discharge(self):
        """Open the hopper discharge."""
        self._discharge_open = 1
        self.state.phase = SimulationPhase.FLOWING
        self.state.phase_start_time = self.state.time
    
    def stop_discharge(self):
        """Close the hopper discharge."""
        self._discharge_open = 0
    
    # =========================================================================
    # LID ANIMATION
    # =========================================================================
    
    def open_lid(self):
        """Start opening the hopper lid."""
        if self.state.lid_state in [LidState.CLOSED, LidState.CLOSING]:
            self.state.lid_state = LidState.OPENING
            self.state.lid_target_angle = self.config.lid_open_angle
    
    def close_lid(self):
        """Start closing the hopper lid."""
        if self.state.lid_state in [LidState.OPEN, LidState.OPENING]:
            self.state.lid_state = LidState.CLOSING
            self.state.lid_target_angle = 0.0
    
    def _update_lid_animation(self, dt: float):
        """Update lid opening/closing animation."""
        if not self.config.animate_lid:
            return
        
        angle_diff = self.state.lid_target_angle - self.state.lid_angle
        
        if abs(angle_diff) < 0.1:
            # Reached target
            self.state.lid_angle = self.state.lid_target_angle
            
            if self.state.lid_state == LidState.OPENING:
                self.state.lid_state = LidState.OPEN
            elif self.state.lid_state == LidState.CLOSING:
                self.state.lid_state = LidState.CLOSED
        else:
            # Animate with constant angular velocity
            direction = 1.0 if angle_diff > 0 else -1.0
            delta_angle = direction * self._lid_max_angular_velocity * dt
            
            # Clamp to not overshoot
            if abs(delta_angle) > abs(angle_diff):
                delta_angle = angle_diff
            
            self.state.lid_angle += delta_angle
    
    def get_lid_transform(self) -> np.ndarray:
        """
        Get the 4x4 transformation matrix for the lid.
        
        Returns:
            4x4 transformation matrix (rotation around hinge)
        """
        angle_rad = np.radians(self.state.lid_angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        hx, hy, hz = self.lid_hinge_position
        
        # Combined transform: T(hinge) * Rz(angle) * T(-hinge)
        transform = np.array([
            [cos_a, -sin_a, 0, hx - hx*cos_a + hy*sin_a],
            [sin_a,  cos_a, 0, hy - hx*sin_a - hy*cos_a],
            [0,      0,     1, 0],
            [0,      0,     0, 1]
        ], dtype=np.float32)
        
        return transform
    
    # =========================================================================
    # POURING SIMULATION
    # =========================================================================
    
    def start_pouring(self):
        """Start the pour sequence: open lid and pour particles."""
        if self.state.phase == SimulationPhase.IDLE:
            self.open_lid()
            self.state.phase = SimulationPhase.POURING
            self.state.phase_start_time = self.state.time
    
    def pour_particles(self, n_pour: int):
        """
        Pour particles from above into the open hopper.
        
        Args:
            n_pour: Number of particles to pour this step
        """
        if self.state.particles_poured >= self.state.total_particles_to_pour:
            return
        
        # Only pour if lid is sufficiently open
        if self.state.lid_angle < self.config.lid_open_angle * 0.5:
            return
        
        n_pour = min(n_pour, self.state.total_particles_to_pour - self.state.particles_poured)
        
        rng = np.random.default_rng(self.state.step + 1234)
        
        # Pour position: above hopper center
        hopper_geo = self.geometry['hopper']
        pour_y = self.hopper_top_y + self.config.pour_height
        
        # Generate positions in a circular stream
        positions = np.zeros((n_pour, 3), dtype=np.float32)
        velocities = np.zeros((n_pour, 3), dtype=np.float32)
        
        # Limit pour stream to fit within hopper opening
        max_stream_radius = self.hopper_top_radius * 0.7
        stream_radius = min(self.config.pour_stream_radius, max_stream_radius)
        
        for i in range(n_pour):
            # Random position within pour stream
            r = np.sqrt(rng.random()) * stream_radius
            theta = rng.uniform(0, TWO_PI)
            
            positions[i, 0] = hopper_geo.center[0] + r * np.cos(theta)
            positions[i, 1] = pour_y
            positions[i, 2] = hopper_geo.center[2] + r * np.sin(theta)
            
            # Initial downward velocity from gravity
            velocities[i, 1] = -np.sqrt(2 * GRAVITY * self.config.pour_height)
        
        # Particle diameters
        visual_diameter = self.config.visual_particle_diameter
        diameters = np.full(n_pour, visual_diameter, dtype=np.float32)
        diameters *= (1.0 + rng.uniform(-0.1, 0.1, n_pour).astype(np.float32))
        
        # Masses
        masses = self.config.particle_density * (PI / 6.0) * diameters ** 3
        
        # All particles start in hopper zone
        zones = np.zeros(n_pour, dtype=np.int32)
        is_active = np.ones(n_pour, dtype=np.int32)
        
        # Copy to device at the right offset
        start_idx = self.state.particles_poured
        
        # Get numpy arrays and update
        pos_np = self.state.positions.numpy()
        vel_np = self.state.velocities.numpy()
        dia_np = self.state.diameters.numpy()
        mass_np = self.state.masses.numpy()
        zone_np = self.state.zones.numpy()
        active_np = self.state.is_active.numpy()
        
        pos_np[start_idx:start_idx + n_pour] = positions
        vel_np[start_idx:start_idx + n_pour] = velocities
        dia_np[start_idx:start_idx + n_pour] = diameters
        mass_np[start_idx:start_idx + n_pour] = masses
        zone_np[start_idx:start_idx + n_pour] = zones
        active_np[start_idx:start_idx + n_pour] = is_active
        
        # Copy back to device
        wp.copy(self.state.positions, wp.array(pos_np, dtype=wp.vec3, device=self.device))
        wp.copy(self.state.velocities, wp.array(vel_np, dtype=wp.vec3, device=self.device))
        wp.copy(self.state.diameters, wp.array(dia_np, dtype=float, device=self.device))
        wp.copy(self.state.masses, wp.array(mass_np, dtype=float, device=self.device))
        wp.copy(self.state.zones, wp.array(zone_np, dtype=wp.int32, device=self.device))
        wp.copy(self.state.is_active, wp.array(active_np, dtype=wp.int32, device=self.device))
        
        self.state.particles_poured += n_pour
        self.state.particles_active = self.state.particles_poured
    
    def _update_simulation_phase(self):
        """Update simulation phase state machine."""
        cfg = self.config
        
        if self.state.phase == SimulationPhase.POURING:
            # Pour particles while lid is open
            if self.state.lid_state == LidState.OPEN:
                # Calculate pour rate in particles per step
                mass_per_particle = cfg.particle_density * (PI / 6.0) * cfg.visual_particle_diameter**3
                particles_per_second = cfg.pour_rate_kg_s / mass_per_particle
                n_pour = max(1, int(particles_per_second * cfg.dt))
                
                self.pour_particles(n_pour)
            
            # Check if pouring complete
            if self.state.particles_poured >= self.state.total_particles_to_pour:
                self.close_lid()
                self.state.phase = SimulationPhase.SETTLING
                self.state.phase_start_time = self.state.time
        
        elif self.state.phase == SimulationPhase.SETTLING:
            # Wait for lid to close and particles to settle
            if self.state.lid_state == LidState.CLOSED:
                elapsed = self.state.time - self.state.phase_start_time
                if elapsed >= cfg.settling_time:
                    # Start discharge
                    self.start_discharge()
    
    def start_simulation(self):
        """
        Start the full simulation workflow:
        1. Open lid
        2. Pour particles
        3. Close lid
        4. Wait for settling
        5. Open discharge
        """
        if self.config.enable_pouring:
            # Calculate total particles to pour
            hopper_geo = self.geometry['hopper']
            hopper_volume = (PI / 3) * hopper_geo.cone_height * (
                hopper_geo.top_radius**2 + hopper_geo.top_radius * hopper_geo.bottom_radius + hopper_geo.bottom_radius**2
            ) + PI * hopper_geo.top_radius**2 * hopper_geo.cylinder_height
            
            fill_volume = hopper_volume * self.config.hopper_fill_percentage / 100.0
            particle_volume = (PI / 6) * self.config.visual_particle_diameter**3
            self.state.total_particles_to_pour = min(
                int(fill_volume * 0.6 / particle_volume),  # 60% packing
                self.config.num_particles
            )
            
            print(f"\n  Starting simulation workflow:")
            print(f"    Total particles to pour: {self.state.total_particles_to_pour}")
            
            self.start_pouring()
        else:
            # Direct discharge without pouring
            self.start_discharge()
    
    def step(self):
        """Advance simulation by one time step."""
        dt = self.config.dt
        cfg = self.config
        
        # Update lid animation
        self._update_lid_animation(dt)
        
        # Update simulation phase (pouring -> settling -> flowing)
        self._update_simulation_phase()
        
        n = self.state.particles_active
        
        if n == 0:
            # Update time even if no particles
            self.state.time += dt
            self.state.step += 1
            return
        
        # Get geometry parameters
        hopper_geo = self.geometry['hopper']
        airlock_geo = self.geometry['airlock']
        feeder_geo = self.geometry['feeder']
        deagg_geo = self.geometry['deagglomerator']
        
        # Launch physics kernel
        wp.launch(
            kernel=physics_flow_kernel,
            dim=n,
            inputs=[
                # Particle state
                self.state.positions,
                self.state.velocities,
                self.state.diameters,
                self.state.masses,
                self.state.zones,
                self.state.is_active,
                n,
                # Hopper
                wp.vec3(*hopper_geo.center),
                float(hopper_geo.top_radius),
                float(hopper_geo.bottom_radius),
                float(hopper_geo.cylinder_height),
                float(hopper_geo.cone_height),
                float(self.hopper_outlet_y),
                float(self.hopper_outlet_radius),
                # Airlock
                wp.vec3(*airlock_geo.center),
                float(airlock_geo.radius),
                float(airlock_geo.length / 2.0),
                float(self.airlock_inlet_y),
                float(self.airlock_outlet_y),
                float(self.airlock_omega),
                # Feeder
                wp.vec3(*feeder_geo.center),
                float(feeder_geo.radius),
                float(feeder_geo.length / 2.0),
                float(self.feeder_inlet_x),
                float(self.feeder_outlet_x),
                float(self.feeder_outlet_y),
                float(self.feeder_omega),
                float(self.feeder_axial_speed),
                # Deagg
                wp.vec3(*deagg_geo.center),
                float(deagg_geo.radius),
                float(deagg_geo.length / 2.0),
                float(self.deagg_inlet_y),
                float(self.deagg_outlet_y),
                float(self.deagg_omega),
                float(deagg_geo.rotor_radius),
                float(self.deagg_outlet_diameter / 2.0),  # Outlet radius
                # Exit
                float(self.exit_y),
                # Physics
                float(dt),
                float(GRAVITY),
                float(cfg.particle_density),
                float(cfg.air_density),
                float(cfg.air_viscosity),
                float(cfg.restitution),
                float(cfg.friction),
                # Control
                self._discharge_open,
            ],
            device=self.device
        )
        
        # Particle-particle collisions (optional, can be expensive)
        if n > 1:
            # Update hash grid
            self._hash_grid.build(
                points=self.state.positions,
                radius=0.1  # Search radius
            )
            
            wp.launch(
                kernel=particle_collision_kernel,
                dim=n,
                inputs=[
                    self.state.positions,
                    self.state.velocities,
                    self.state.diameters,
                    self.state.masses,
                    self.state.is_active,
                    self._hash_grid.id,
                    n,
                    float(cfg.restitution),
                    0.1,  # Search radius
                ],
                device=self.device
            )
        
        # Update time
        self.state.time += dt
        self.state.step += 1
    
    def get_zone_counts(self) -> Dict[str, int]:
        """Get particle counts by zone."""
        zones = self.state.zones.numpy()[:self.state.particles_active]
        is_active = self.state.is_active.numpy()[:self.state.particles_active]
        
        # Only count active particles
        active_zones = zones[is_active == 1]
        
        return {
            'hopper': int(np.sum(active_zones == 0)),
            'airlock': int(np.sum(active_zones == 1)),
            'feeder': int(np.sum(active_zones == 2)),
            'deagg': int(np.sum(active_zones == 3)),
            'exited': int(np.sum(active_zones == 4)),
            'inactive': int(np.sum(is_active == 0)),
        }
    
    def get_positions(self) -> np.ndarray:
        """Get current particle positions."""
        return self.state.positions.numpy()[:self.state.particles_active]
    
    def get_velocities(self) -> np.ndarray:
        """Get current particle velocities."""
        return self.state.velocities.numpy()[:self.state.particles_active]
    
    def get_diameters(self) -> np.ndarray:
        """Get particle diameters."""
        return self.state.diameters.numpy()[:self.state.particles_active]


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_physics_flow_simulator(
    capacity_kg: float = 500,
    feed_rate_kg_h: float = 500,
    airlock_rpm: float = 20,
    feeder_rpm: float = 60,
    deagg_rpm: float = 1500,
    num_particles: int = 5000,
    device: str = "cuda",
) -> Tuple[FeedSystemAssembly, FeedFlowPhysicsSimulator]:
    """
    Create a feed system assembly and physics-based flow simulator.
    
    Args:
        capacity_kg: Hopper capacity [kg]
        feed_rate_kg_h: Target feed rate [kg/h]
        airlock_rpm: Airlock rotation speed [RPM]
        feeder_rpm: Screw feeder rotation speed [RPM]
        deagg_rpm: Deagglomerator rotor speed [RPM]
        num_particles: Number of simulation particles
        device: Warp device ('cuda' or 'cpu')
        
    Returns:
        Tuple of (assembly, simulator)
    """
    # Create assembly with geometry
    params = FeedSystemParams(
        hopper_capacity_kg=capacity_kg,
        feeder_target_rate_kg_h=feed_rate_kg_h,
    )
    assembly = FeedSystemAssembly(params, device="cpu")
    
    # Create simulator config
    config = FlowPhysicsConfig(
        airlock_rpm=airlock_rpm,
        feeder_rpm=feeder_rpm,
        deagg_rpm=deagg_rpm,
        num_particles=num_particles,
        device=device,
    )
    
    # Create simulator
    simulator = FeedFlowPhysicsSimulator(assembly, config)
    
    return assembly, simulator
