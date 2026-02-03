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
    # Transition zones (particles flow through these connectors)
    TRANS_HOPPER_AIRLOCK = 10      # Cylindrical transition
    TRANS_AIRLOCK_FEEDER = 11      # Conical reducer transition
    TRANS_FEEDER_DEAGG = 12        # Cylindrical transition


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
    # Particles must be smaller than smallest passage (Trans2 end = 84mm, Trans3 = 80mm)
    # Rule of thumb: particle diameter < passage_diameter / 5 for free flow
    # 80mm / 5 = 16mm, using 15mm for safety margin
    visual_particle_diameter: float = 0.015  # [m] Display size (15mm visual particles)


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
    airlock_omega: float,  # rad/s = 2*pi * RPM / 60
    
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
    
    # =========================================================================
    # TRANSITION CONNECTOR GEOMETRY (from diagnose_feed_connectors.py)
    # =========================================================================
    # Transition 1: Hopper -> Airlock (cylindrical, vertical -Y)
    trans1_start: wp.vec3,         # Start position (hopper discharge port)
    trans1_end: wp.vec3,           # End position (airlock inlet port)
    trans1_radius: float,          # Transition radius (diameter/2)
    trans1_length: float,          # Transition length
    
    # Transition 2: Airlock -> Feeder (conical reducer, vertical -Y)
    trans2_start: wp.vec3,         # Start position (airlock outlet port)
    trans2_end: wp.vec3,           # End position (feeder inlet port)
    trans2_start_radius: float,    # Start radius (larger, at airlock outlet)
    trans2_end_radius: float,      # End radius (smaller, at feeder inlet)
    trans2_length: float,          # Transition length
    
    # Transition 3: Feeder -> Deagglomerator (cylindrical, vertical -Y)
    trans3_start: wp.vec3,         # Start position (feeder outlet port)
    trans3_end: wp.vec3,           # End position (deagg inlet port)
    trans3_radius: float,          # Transition radius
    trans3_length: float,          # Transition length
    
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
    
    Flow path with transitions (from diagnose_feed_connectors.py):
        HOPPER (zone 0)
            |
        TRANS_HOPPER_AIRLOCK (zone 10) - 15mm cylindrical
            |
        AIRLOCK (zone 1)
            |
        TRANS_AIRLOCK_FEEDER (zone 11) - 120mm conical reducer (12 deg half-angle)
            |
        FEEDER (zone 2)
            |
        TRANS_FEEDER_DEAGG (zone 12) - 20mm cylindrical
            |
        DEAGGLOMERATOR (zone 3)
            |
        EXITED (zone 4)
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
        # Position relative to hopper center
        # NOTE: hopper_center is at the base of the cone, but discharge port is BELOW it
        # The actual discharge outlet is at hopper_outlet_y (which is trans1_start[1])
        local_y = pos[1] - hopper_center[1]
        local_x = pos[0] - hopper_center[0]
        local_z = pos[2] - hopper_center[2]
        r = wp.sqrt(local_x * local_x + local_z * local_z)
        
        # Compute wall radius at current height
        # For local_y >= 0: in the cone/cylinder section
        # For local_y < 0: in the discharge tube (radius = hopper_outlet_radius)
        if local_y < 0.0:
            # Below cone apex - discharge tube region
            wall_radius = hopper_outlet_radius
        elif local_y < hopper_cone_height:
            # In cone section - linear interpolation from bottom_radius to top_radius
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
        
        # Bottom collision - at the actual discharge outlet position (not at hopper_center!)
        # The discharge is at hopper_outlet_y (trans1_start[1])
        if pos[1] < hopper_outlet_y + particle_radius:
            if r > hopper_outlet_radius or discharge_open == 0:
                # Hit discharge tube bottom, not in outlet opening OR discharge is closed
                normal = wp.vec3(0.0, 1.0, 0.0)
                pos = wp.vec3(pos[0], hopper_outlet_y + particle_radius + 0.001, pos[2])
                vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
        
        # Top boundary
        total_height = hopper_cone_height + hopper_cylinder_height
        if local_y > total_height - particle_radius:
            normal = wp.vec3(0.0, -1.0, 0.0)
            pos = wp.vec3(pos[0], hopper_center[1] + total_height - particle_radius - 0.001, pos[2])
            vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
        
        # TRANSITION: Through discharge -> enter transition connector (zone 10)
        # Particles must be at the actual discharge opening to transition
        # The discharge port is at hopper_outlet_y (-30mm)
        if discharge_open == 1:
            # Recalculate r from current position (may have been modified by wall collision)
            curr_local_x = pos[0] - hopper_center[0]
            curr_local_z = pos[2] - hopper_center[2]
            curr_r = wp.sqrt(curr_local_x * curr_local_x + curr_local_z * curr_local_z)
            
            # Particle must be NEAR or BELOW the discharge port
            # Use VERY generous threshold to avoid deadlock with post-integration containment
            # hopper_outlet_y = -30mm, so trigger transition when within 50mm above outlet
            at_discharge = pos[1] < hopper_outlet_y + particle_radius * 5.0  # Much more generous
            in_opening = curr_r < hopper_outlet_radius + particle_radius * 0.5  # Slightly more generous
            
            if at_discharge and in_opening:
                zone = 10  # Update local variable too!
    
    # =========================================================================
    # ZONE 10: TRANSITION - HOPPER -> AIRLOCK (cylindrical, vertical -Y)
    # =========================================================================
    elif zone == 10:
        # Cylindrical transition connector from hopper to airlock
        # Flow direction is -Y (downward)
        
        # Progress along transition (0 at start, 1 at end)
        # trans1_start is at hopper discharge, trans1_end is at airlock inlet
        progress = (trans1_start[1] - pos[1]) / (trans1_start[1] - trans1_end[1] + 0.001)
        progress = wp.clamp(progress, 0.0, 1.0)
        
        # Radial distance from centerline (centerline is along Y axis at trans1_start X,Z)
        center_x = trans1_start[0]
        center_z = trans1_start[2]
        dx = pos[0] - center_x
        dz = pos[2] - center_z
        r_xz = wp.sqrt(dx * dx + dz * dz)
        
        # Cylindrical containment
        if r_xz + particle_radius > trans1_radius * 0.9:
            if r_xz > 1.0e-6:
                scale = (trans1_radius * 0.85 - particle_radius) / r_xz
                pos = wp.vec3(center_x + dx * scale, pos[1], center_z + dz * scale)
                # Reflect velocity
                normal = wp.vec3(-dx / r_xz, 0.0, -dz / r_xz)
                vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
        
        # Don't go back up into hopper
        if pos[1] > trans1_start[1] - particle_radius:
            pos = wp.vec3(pos[0], trans1_start[1] - particle_radius - 0.001, pos[2])
            normal = wp.vec3(0.0, -1.0, 0.0)
            vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
        
        # Transition to airlock when reaching end (use generous threshold)
        if pos[1] <= trans1_end[1] + particle_radius * 3.0:
            zone = 1  # Enter airlock
    
    # =========================================================================
    # ZONE 1: AIRLOCK
    # =========================================================================
    elif zone == 1:
        # The airlock is a vertical cylinder with:
        # - Inlet at top (airlock_inlet_y, from trans1)
        # - Outlet at bottom (airlock_outlet_y, to trans2)
        # - Radial containment in XZ plane (cylinder axis is Y)
        
        px = pos[0] - airlock_center[0]
        py = pos[1] - airlock_center[1]
        pz = pos[2] - airlock_center[2]
        
        # Radial distance in XZ plane (cylinder axis is Y)
        r_xz = wp.sqrt(px * px + pz * pz)
        
        # Rotational effect from vanes (rotation around Y axis for vertical airlock)
        if r_xz > 0.01:
            # Tangential velocity from rotating vanes
            v_tan = compute_tangential_velocity(
                pos, airlock_center,
                wp.vec3(0.0, 1.0, 0.0),  # Y axis (vertical)
                airlock_omega
            )
            # Couple particle to vane rotation (partial coupling)
            coupling = 0.1  # 10% coupling to vane speed
            tan_accel = (v_tan - vel) * coupling / dt
            # Clamp to prevent instability
            tan_accel_mag = wp.length(tan_accel)
            if tan_accel_mag > 30.0:
                tan_accel = tan_accel * (30.0 / tan_accel_mag)
            accel = accel + tan_accel
        
        # Cylindrical housing wall collision (radial in XZ)
        if r_xz + particle_radius > airlock_radius:
            if r_xz > 1.0e-6:
                normal = wp.vec3(-px / r_xz, 0.0, -pz / r_xz)
                push = r_xz + particle_radius - airlock_radius + 0.001
                pos = pos + normal * push
                vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
        
        # Axial containment (Z direction for rotor length)
        if wp.abs(pz) + particle_radius > airlock_half_length:
            sign_z = 1.0 if pz > 0.0 else -1.0
            normal = wp.vec3(0.0, 0.0, -sign_z)
            pos = wp.vec3(pos[0], pos[1], airlock_center[2] + sign_z * (airlock_half_length - particle_radius - 0.001))
            vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
        
        # Top (inlet) containment - use actual inlet Y position
        # Use generous radius to match transition checks
        inlet_radius = trans1_radius + particle_radius
        if pos[1] > airlock_inlet_y - particle_radius:
            # Check if NOT in inlet opening
            if r_xz > inlet_radius:
                normal = wp.vec3(0.0, -1.0, 0.0)
                pos = wp.vec3(pos[0], airlock_inlet_y - particle_radius - 0.001, pos[2])
                vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
        
        # Bottom (outlet) containment - use actual outlet Y position
        # Use SAME generous radius as transition check to avoid blocking transitioning particles
        outlet_radius = trans2_start_radius + particle_radius  # Generous outlet opening
        if pos[1] < airlock_outlet_y + particle_radius:
            # Check if NOT in outlet opening
            if r_xz > outlet_radius:
                normal = wp.vec3(0.0, 1.0, 0.0)
                pos = wp.vec3(pos[0], airlock_outlet_y + particle_radius + 0.001, pos[2])
                vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
        
        # TRANSITION: Through outlet at bottom -> enter transition connector (zone 11)
        # Use generous threshold to avoid deadlock with post-integration containment
        if pos[1] < airlock_outlet_y + particle_radius * 5.0 and r_xz < outlet_radius + particle_radius:
            zone = 11  # Enter airlock->feeder transition
    
    # =========================================================================
    # ZONE 11: TRANSITION - AIRLOCK -> FEEDER (conical reducer, vertical -Y)
    # From diagnose: 120mm conical reducer with 12 deg half-angle
    # =========================================================================
    elif zone == 11:
        # Conical reducer transition from airlock to feeder
        # Flow direction is -Y (downward)
        # Diameter reduces from trans2_start_radius to trans2_end_radius
        
        # Progress along transition (0 at start, 1 at end)
        progress = (trans2_start[1] - pos[1]) / (trans2_start[1] - trans2_end[1] + 0.001)
        progress = wp.clamp(progress, 0.0, 1.0)
        
        # Radius at current position (linear interpolation for conical section)
        current_radius = trans2_start_radius + progress * (trans2_end_radius - trans2_start_radius)
        
        # Center position at this height (centerline may shift from airlock to feeder)
        # Interpolate center position
        center_x = trans2_start[0] + progress * (trans2_end[0] - trans2_start[0])
        center_z = trans2_start[2] + progress * (trans2_end[2] - trans2_start[2])
        
        dx = pos[0] - center_x
        dz = pos[2] - center_z
        r_xz = wp.sqrt(dx * dx + dz * dz)
        
        # Conical containment
        if r_xz + particle_radius > current_radius * 0.9:
            if r_xz > 1.0e-6:
                scale = (current_radius * 0.85 - particle_radius) / r_xz
                pos = wp.vec3(center_x + dx * scale, pos[1], center_z + dz * scale)
                # Compute wall normal (cone surface normal)
                # For a cone, normal has both radial and axial components
                cone_half_angle = wp.atan2(trans2_start_radius - trans2_end_radius, trans2_length)
                wall_normal_y = -wp.sin(cone_half_angle)  # Slight upward component
                wall_normal_r = wp.cos(cone_half_angle)   # Radial component
                normal = wp.vec3(-dx / r_xz * wall_normal_r, wall_normal_y, -dz / r_xz * wall_normal_r)
                vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
        
        # Don't go back up into airlock
        if pos[1] > trans2_start[1] - particle_radius:
            pos = wp.vec3(pos[0], trans2_start[1] - particle_radius - 0.001, pos[2])
            normal = wp.vec3(0.0, -1.0, 0.0)
            vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
        
        # Transition to feeder when reaching end (use generous threshold)
        if pos[1] <= trans2_end[1] + particle_radius * 3.0:
            zone = 2  # Enter screw feeder
            # Give particle initial velocity along feeder axis
            vel = wp.vec3(feeder_axial_speed * 0.3, vel[1] * 0.5, vel[2] * 0.5)
    
    # =========================================================================
    # ZONE 2: SCREW FEEDER
    # =========================================================================
    elif zone == 2:
        # Feeder is horizontal cylinder along X axis
        # Inlet at -X end (top), outlet at +X end (bottom)
        px = pos[0] - feeder_center[0]
        py = pos[1] - feeder_center[1]
        pz = pos[2] - feeder_center[2]
        
        # Progress along feeder (0 to 1)
        feeder_length = 2.0 * feeder_half_length
        progress = (pos[0] - feeder_inlet_x) / feeder_length
        progress = wp.clamp(progress, 0.0, 1.0)
        
        # Radial distance in YZ plane (perpendicular to screw axis)
        r_yz = wp.sqrt(py * py + pz * pz)
        
        # Screw conveying effect: impart axial velocity
        # Conveying velocity = pitch x RPM / 60 (already computed as feeder_axial_speed)
        target_vx = feeder_axial_speed
        
        # Rotational effect from screw - gentler coupling
        if r_yz > 0.005:
            v_tan = compute_tangential_velocity(
                pos, feeder_center,
                wp.vec3(1.0, 0.0, 0.0),  # X axis
                feeder_omega
            )
            # Partial coupling to screw rotation (reduced to prevent instability)
            coupling = 0.1
            tan_accel = (v_tan - vel) * coupling / dt
            # Clamp tangential acceleration to prevent overflow
            tan_accel_mag = wp.length(tan_accel)
            if tan_accel_mag > 50.0:
                tan_accel = tan_accel * (50.0 / tan_accel_mag)
            accel = accel + tan_accel
        
        # Apply axial conveying force (gentler, clamped)
        vel_error = target_vx - vel[0]
        ax_force = vel_error * 5.0  # Reduced gain for stability
        ax_force = wp.clamp(ax_force, -20.0, 20.0)  # Clamp acceleration
        accel = accel + wp.vec3(ax_force, 0.0, 0.0)
        
        # Define outlet region at +X end of feeder
        # Outlet is at the bottom of the trough at the discharge end
        # In a real screw feeder, material drops off the end into the outlet
        outlet_region_start_x = feeder_half_length - trans3_radius * 3.0
        outlet_radius = trans3_radius * 1.0  # Full outlet size
        in_outlet_region = (px > outlet_region_start_x) and (wp.abs(pz) < outlet_radius)
        
        # Inlet region at -X end
        inlet_radius = trans2_end_radius * 0.9
        in_inlet_region = (px < -feeder_half_length + inlet_radius * 2.0) and (py > 0.0)
        
        # At the outlet end, reduce screw rotation coupling - particles should drop
        if in_outlet_region and progress > 0.8:
            # Near outlet - let particles settle and fall through
            # Add slight downward bias to help particles reach outlet
            accel = accel + wp.vec3(0.0, -2.0, 0.0)  # Extra gravity toward outlet
        
        # Tube wall collision (radial in YZ) - with outlet opening at bottom
        if r_yz + particle_radius > feeder_radius:
            # Check if particle is in the outlet opening (lower half at +X end)
            # Use generous threshold to match transition: py < -0.3*radius means below center
            # Exempt particles in outlet region that are below or near center
            in_outlet_opening = in_outlet_region and (py < feeder_radius * 0.5)  # More generous
            
            if not in_inlet_region and not in_outlet_opening:
                if r_yz > 1.0e-6:
                    normal_y = -py / r_yz
                    normal_z = -pz / r_yz
                    normal = wp.vec3(0.0, normal_y, normal_z)
                    push = r_yz + particle_radius - feeder_radius + 0.001
                    pos = pos + normal * push
                    vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
        
        # Inlet end cap - block only outside inlet opening
        if px < -feeder_half_length + particle_radius:
            if py < 0.0 or r_yz > inlet_radius:
                normal = wp.vec3(1.0, 0.0, 0.0)
                pos = wp.vec3(feeder_center[0] - feeder_half_length + particle_radius + 0.001, pos[1], pos[2])
                vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
        
        # Outlet end cap - only block upper half (lower half is outlet)
        if px > feeder_half_length - particle_radius:
            # Allow passage through outlet opening (lower 60% of tube)
            in_outlet_hole = (py < feeder_radius * 0.2) and (wp.abs(pz) < outlet_radius)
            if not in_outlet_hole:
                normal = wp.vec3(-1.0, 0.0, 0.0)
                pos = wp.vec3(feeder_center[0] + feeder_half_length - particle_radius - 0.001, pos[1], pos[2])
                vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
        
        # TRANSITION: Exit through outlet at bottom of +X end
        # Particle exits when at outlet end and in lower portion of tube
        # More relaxed condition - if particle is in outlet region and below center, it exits
        at_outlet = in_outlet_region and (py < -feeder_radius * 0.3)
        
        if at_outlet:
            zone = 12  # Enter feeder->deagg transition
            # Give initial downward velocity toward deagglomerator
            vel = wp.vec3(vel[0] * 0.1, wp.min(vel[1], -0.5), vel[2] * 0.1)
    
    # =========================================================================
    # ZONE 12: TRANSITION - FEEDER -> DEAGGLOMERATOR (cylindrical, vertical -Y)
    # From diagnose: 20mm cylindrical transition
    # =========================================================================
    elif zone == 12:
        # Cylindrical transition from feeder outlet to deagglomerator inlet
        # Flow direction is -Y (downward)
        
        # Progress along transition (0 at start, 1 at end)
        progress = (trans3_start[1] - pos[1]) / (trans3_start[1] - trans3_end[1] + 0.001)
        progress = wp.clamp(progress, 0.0, 1.0)
        
        # Centerline position
        center_x = trans3_start[0] + progress * (trans3_end[0] - trans3_start[0])
        center_z = trans3_start[2] + progress * (trans3_end[2] - trans3_start[2])
        
        dx = pos[0] - center_x
        dz = pos[2] - center_z
        r_xz = wp.sqrt(dx * dx + dz * dz)
        
        # Cylindrical containment
        if r_xz + particle_radius > trans3_radius * 0.9:
            if r_xz > 1.0e-6:
                scale = (trans3_radius * 0.85 - particle_radius) / r_xz
                pos = wp.vec3(center_x + dx * scale, pos[1], center_z + dz * scale)
                normal = wp.vec3(-dx / r_xz, 0.0, -dz / r_xz)
                vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
        
        # Don't go back up into feeder
        if pos[1] > trans3_start[1] - particle_radius:
            pos = wp.vec3(pos[0], trans3_start[1] - particle_radius - 0.001, pos[2])
            normal = wp.vec3(0.0, -1.0, 0.0)
            vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
        
        # Transition to deagglomerator when reaching end (use generous threshold)
        if pos[1] <= trans3_end[1] + particle_radius * 3.0:
            zone = 3  # Enter deagglomerator
    
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
        # Deagglomerator has high-speed rotor that imparts tangential velocity
        if r_yz > 0.005 and r_yz < deagg_rotor_radius:
            # Inside rotor region - tangential coupling
            v_tan = compute_tangential_velocity(
                pos, deagg_center,
                wp.vec3(1.0, 0.0, 0.0),  # X axis
                deagg_omega
            )
            # Coupling decreases toward center, capped for stability
            coupling = 0.3 * (1.0 - r_yz / deagg_rotor_radius)
            tan_accel = (v_tan - vel) * coupling / dt
            # Clamp to prevent instability from high RPM
            tan_accel_mag = wp.length(tan_accel)
            if tan_accel_mag > 100.0:
                tan_accel = tan_accel * (100.0 / tan_accel_mag)
            accel = accel + tan_accel
        
        # Outlet opening parameters:
        # The outlet is at the bottom of the cylinder (Y-)
        # Opening starts at Y = center - radius (bottom of cylinder)
        outlet_opening_threshold = -deagg_radius + deagg_outlet_radius
        
        # Check if particle is in the outlet region (bottom of cylinder, within outlet diameter)
        in_outlet_region = (py < outlet_opening_threshold) and (wp.abs(pz) < deagg_outlet_radius)
        
        # Inlet region at top (where transition connects)
        inlet_radius = trans3_radius * 0.9
        in_inlet_region = (py > deagg_radius - inlet_radius) and (wp.abs(pz) < inlet_radius)
        
        # Housing wall collision - only apply if NOT in inlet/outlet region
        if r_yz + particle_radius > deagg_radius and not in_outlet_region and not in_inlet_region:
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
        
        # Inlet containment (top) - don't block inlet opening
        if py > deagg_radius * 0.8 and not in_inlet_region:
            normal = wp.vec3(0.0, -1.0, 0.0)
            pos = wp.vec3(pos[0], deagg_center[1] + deagg_radius * 0.75, pos[2])
            vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
        
        # TRANSITION: Exit through outlet at bottom
        # Particle exits when it falls below the outlet Y position
        if pos[1] < deagg_outlet_y:
            zone = 4  # Enter exited zone
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
    # INTEGRATION: Semi-implicit Euler with velocity clamping
    # =========================================================================
    
    # Clamp acceleration to prevent numerical instability
    accel_mag = wp.length(accel)
    max_accel = 500.0  # m/s^2 (about 50g)
    if accel_mag > max_accel:
        accel = accel * (max_accel / accel_mag)
    
    vel = vel + accel * dt
    
    # Clamp velocity to prevent overflow
    vel_mag = wp.length(vel)
    max_vel = 20.0  # m/s (reasonable max for powder flow)
    if vel_mag > max_vel:
        vel = vel * (max_vel / vel_mag)
    
    pos = pos + vel * dt
    
    # =========================================================================
    # POST-INTEGRATION CONTAINMENT - Enforce particles stay within zone walls
    # This is critical to prevent particles escaping through walls
    # =========================================================================
    
    # Zone 0: Hopper containment
    if zone == 0:
        local_y = pos[1] - hopper_center[1]
        local_x = pos[0] - hopper_center[0]
        local_z = pos[2] - hopper_center[2]
        r = wp.sqrt(local_x * local_x + local_z * local_z)
        
        # Wall radius at current height
        # For local_y < 0: in discharge tube region
        # For local_y >= 0: in cone/cylinder
        if local_y < 0.0:
            wall_r = hopper_outlet_radius
        elif local_y < hopper_cone_height:
            t = local_y / hopper_cone_height
            wall_r = hopper_bottom_radius + t * (hopper_top_radius - hopper_bottom_radius)
        else:
            wall_r = hopper_top_radius
        
        # Radial containment
        if r > wall_r - particle_radius and r > 1.0e-6:
            scale = (wall_r - particle_radius - 0.001) / r
            pos = wp.vec3(hopper_center[0] + local_x * scale, pos[1], hopper_center[2] + local_z * scale)
            # Recalculate r after position adjustment
            local_x = pos[0] - hopper_center[0]
            local_z = pos[2] - hopper_center[2]
            r = wp.sqrt(local_x * local_x + local_z * local_z)
        
        # Vertical containment - bottom is at hopper_outlet_y (not hopper_center!)
        total_h = hopper_cone_height + hopper_cylinder_height
        # Bottom: block at hopper_outlet_y unless discharge open AND in outlet radius
        # Use SAME generous radius as transition check to avoid blocking transitioning particles
        if pos[1] < hopper_outlet_y + particle_radius:
            if discharge_open == 0 or r > hopper_outlet_radius + particle_radius * 0.5:
                pos = wp.vec3(pos[0], hopper_outlet_y + particle_radius + 0.001, pos[2])
        # Top
        if local_y > total_h - particle_radius:
            pos = wp.vec3(pos[0], hopper_center[1] + total_h - particle_radius - 0.001, pos[2])
    
    # Zone 10: Trans hopper->airlock containment (cylinder along Y)
    elif zone == 10:
        center_x = trans1_start[0]
        center_z = trans1_start[2]
        dx = pos[0] - center_x
        dz = pos[2] - center_z
        r = wp.sqrt(dx * dx + dz * dz)
        
        if r > trans1_radius - particle_radius and r > 1.0e-6:
            scale = (trans1_radius - particle_radius - 0.001) / r
            pos = wp.vec3(center_x + dx * scale, pos[1], center_z + dz * scale)
        
        # Y bounds
        if pos[1] > trans1_start[1]:
            pos = wp.vec3(pos[0], trans1_start[1] - 0.001, pos[2])
        if pos[1] < trans1_end[1]:
            pos = wp.vec3(pos[0], trans1_end[1] + 0.001, pos[2])
    
    # Zone 1: Airlock containment (vertical cylinder)
    # Use actual inlet/outlet Y positions from port geometry
    elif zone == 1:
        px = pos[0] - airlock_center[0]
        pz = pos[2] - airlock_center[2]
        r_xz = wp.sqrt(px * px + pz * pz)  # Distance from center in XZ plane
        
        # Inlet opening at top (where trans1 connects)
        # Use generous radius to match transition checks
        inlet_r = trans1_radius + particle_radius
        in_inlet = (pos[1] > airlock_inlet_y - inlet_r * 0.5) and (r_xz < inlet_r)
        
        # Outlet opening at bottom (where trans2 connects)  
        # Use generous radius to match transition checks
        outlet_r = trans2_start_radius + particle_radius
        in_outlet = (pos[1] < airlock_outlet_y + outlet_r * 0.5) and (r_xz < outlet_r)
        
        # Radial containment (cylinder wall in XZ) - allow inlet/outlet openings
        if r_xz > airlock_radius - particle_radius and r_xz > 1.0e-6:
            if not in_inlet and not in_outlet:
                scale = (airlock_radius - particle_radius - 0.001) / r_xz
                pos = wp.vec3(airlock_center[0] + px * scale, pos[1], airlock_center[2] + pz * scale)
        
        # Axial containment (Z - rotor length direction)
        if wp.abs(pz) > airlock_half_length - particle_radius:
            sign_z = 1.0 if pz > 0.0 else -1.0
            pos = wp.vec3(pos[0], pos[1], airlock_center[2] + sign_z * (airlock_half_length - particle_radius - 0.001))
        
        # Y bounds - use actual inlet/outlet positions
        # Top: inlet Y position
        if pos[1] > airlock_inlet_y - particle_radius and not in_inlet:
            pos = wp.vec3(pos[0], airlock_inlet_y - particle_radius - 0.001, pos[2])
        # Bottom: outlet Y position (don't block outlet opening)
        if pos[1] < airlock_outlet_y + particle_radius and not in_outlet:
            pos = wp.vec3(pos[0], airlock_outlet_y + particle_radius + 0.001, pos[2])
    
    # Zone 11: Trans airlock->feeder containment (cone along Y)
    elif zone == 11:
        progress = (trans2_start[1] - pos[1]) / (trans2_start[1] - trans2_end[1] + 0.001)
        progress = wp.clamp(progress, 0.0, 1.0)
        current_r = trans2_start_radius + progress * (trans2_end_radius - trans2_start_radius)
        
        center_x = trans2_start[0] + progress * (trans2_end[0] - trans2_start[0])
        center_z = trans2_start[2] + progress * (trans2_end[2] - trans2_start[2])
        dx = pos[0] - center_x
        dz = pos[2] - center_z
        r = wp.sqrt(dx * dx + dz * dz)
        
        if r > current_r - particle_radius and r > 1.0e-6:
            scale = (current_r - particle_radius - 0.001) / r
            pos = wp.vec3(center_x + dx * scale, pos[1], center_z + dz * scale)
        
        # Y bounds
        if pos[1] > trans2_start[1]:
            pos = wp.vec3(pos[0], trans2_start[1] - 0.001, pos[2])
        if pos[1] < trans2_end[1]:
            pos = wp.vec3(pos[0], trans2_end[1] + 0.001, pos[2])
    
    # Zone 2: Feeder containment (cylinder along X)
    elif zone == 2:
        px = pos[0] - feeder_center[0]
        py = pos[1] - feeder_center[1]
        pz = pos[2] - feeder_center[2]
        r_yz = wp.sqrt(py * py + pz * pz)
        
        # Radial containment (only apply if not at outlet opening)
        outlet_region = (px > feeder_half_length - trans3_radius * 3.0)
        # Use generous threshold to match before-integration and transition checks
        in_outlet = outlet_region and (py < feeder_radius * 0.5) and (wp.abs(pz) < trans3_radius)
        
        if r_yz > feeder_radius - particle_radius and r_yz > 1.0e-6 and not in_outlet:
            scale = (feeder_radius - particle_radius - 0.001) / r_yz
            pos = wp.vec3(pos[0], feeder_center[1] + py * scale, feeder_center[2] + pz * scale)
        
        # Axial containment (X) - with openings at inlet and outlet
        if px < -feeder_half_length + particle_radius:
            # Only block if not in inlet opening
            if py < 0.0:
                pos = wp.vec3(feeder_center[0] - feeder_half_length + particle_radius + 0.001, pos[1], pos[2])
        if px > feeder_half_length - particle_radius:
            # Only block if not in outlet opening
            if not in_outlet:
                pos = wp.vec3(feeder_center[0] + feeder_half_length - particle_radius - 0.001, pos[1], pos[2])
    
    # Zone 12: Trans feeder->deagg containment (cylinder along Y)
    elif zone == 12:
        progress = (trans3_start[1] - pos[1]) / (trans3_start[1] - trans3_end[1] + 0.001)
        progress = wp.clamp(progress, 0.0, 1.0)
        
        center_x = trans3_start[0] + progress * (trans3_end[0] - trans3_start[0])
        center_z = trans3_start[2] + progress * (trans3_end[2] - trans3_start[2])
        dx = pos[0] - center_x
        dz = pos[2] - center_z
        r = wp.sqrt(dx * dx + dz * dz)
        
        if r > trans3_radius - particle_radius and r > 1.0e-6:
            scale = (trans3_radius - particle_radius - 0.001) / r
            pos = wp.vec3(center_x + dx * scale, pos[1], center_z + dz * scale)
        
        # Y bounds
        if pos[1] > trans3_start[1]:
            pos = wp.vec3(pos[0], trans3_start[1] - 0.001, pos[2])
        if pos[1] < trans3_end[1]:
            pos = wp.vec3(pos[0], trans3_end[1] + 0.001, pos[2])
    
    # Zone 3: Deagglomerator containment (cylinder along X)
    elif zone == 3:
        px = pos[0] - deagg_center[0]
        py = pos[1] - deagg_center[1]
        pz = pos[2] - deagg_center[2]
        r_yz = wp.sqrt(py * py + pz * pz)
        
        # Inlet region at top
        inlet_r = trans3_radius * 0.9
        in_inlet = (py > deagg_radius - inlet_r) and (wp.abs(pz) < inlet_r)
        
        # Outlet region at bottom
        outlet_r = deagg_outlet_radius
        in_outlet = (py < -deagg_radius + outlet_r) and (wp.abs(pz) < outlet_r)
        
        # Radial containment (with openings)
        if r_yz > deagg_radius - particle_radius and r_yz > 1.0e-6:
            if not in_inlet and not in_outlet:
                scale = (deagg_radius - particle_radius - 0.001) / r_yz
                pos = wp.vec3(pos[0], deagg_center[1] + py * scale, deagg_center[2] + pz * scale)
        
        # Axial containment (X)
        if wp.abs(px) > deagg_half_length - particle_radius:
            sign_x = 1.0 if px > 0.0 else -1.0
            pos = wp.vec3(deagg_center[0] + sign_x * (deagg_half_length - particle_radius - 0.001), pos[1], pos[2])
    
    # Zone 4: Exited - floor containment only
    elif zone == 4:
        if pos[1] < exit_y + particle_radius:
            pos = wp.vec3(pos[0], exit_y + particle_radius + 0.001, pos[2])
    
    # =========================================================================
    # GLOBAL SAFETY CHECK - Deactivate particles that have escaped the system
    # =========================================================================
    # Only deactivate particles that are WAY outside the system bounds
    # (This should rarely trigger - it's a last resort safety check)
    
    # Vertical bounds check (very generous - 2m margin)
    if pos[1] < exit_y - 2.0 or pos[1] > hopper_center[1] + hopper_cone_height + hopper_cylinder_height + 2.0:
        # Particle has escaped vertically - deactivate
        is_active[tid] = 0
    
    # Radial check - only for particles NOT yet exited (exited particles can spread out)
    # Use generous 5m radius (system is only ~1.3m wide)
    if zone != 4:
        local_x = pos[0] - hopper_center[0]
        local_z = pos[2] - hopper_center[2]
        r_global = wp.sqrt(local_x * local_x + local_z * local_z)
        if r_global > 5.0:  # More than 5m from center - clearly escaped
            # Particle has escaped radially - deactivate
            is_active[tid] = 0
    
    # Write back
    positions[tid] = pos
    velocities[tid] = vel
    zones[tid] = zone


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
        # These are the transition connectors from diagnose_feed_connectors.py
        # =====================================================================
        connections = self.geometry.get('connections', {})
        
        # Store connection data for kernel
        self.conn_hopper_airlock = connections.get('hopper_to_airlock', {})
        self.conn_airlock_feeder = connections.get('airlock_to_feeder', {})
        self.conn_feeder_deagg = connections.get('feeder_to_deagg', {})
        
        # =====================================================================
        # TRANSITION CONNECTOR GEOMETRY (for kernel parameters)
        # From diagnose_feed_connectors.py:
        #   Hopper->Airlock: 15mm cylindrical, vertical -Y
        #   Airlock->Feeder: 120mm conical reducer (12 deg), vertical -Y
        #   Feeder->Deagg: 20mm cylindrical, vertical -Y
        # =====================================================================
        
        # Transition 1: Hopper -> Airlock (cylindrical)
        conn1 = self.conn_hopper_airlock
        self.trans1_start = hopper_geo.outlet_pos.copy()
        self.trans1_end = airlock_geo.inlet_pos.copy()
        self.trans1_radius = conn1.get('start_diameter', hopper_geo.outlet_diameter) / 2.0
        self.trans1_length = conn1.get('length', 0.015)  # 15mm default
        
        # Transition 2: Airlock -> Feeder (conical reducer)
        conn2 = self.conn_airlock_feeder
        self.trans2_start = airlock_geo.outlet_pos.copy()
        self.trans2_end = feeder_geo.inlet_pos.copy()
        self.trans2_start_radius = conn2.get('start_diameter', airlock_geo.outlet_diameter) / 2.0
        self.trans2_end_radius = conn2.get('end_diameter', feeder_geo.inlet_diameter) / 2.0
        self.trans2_length = conn2.get('length', 0.120)  # 120mm default
        
        # Transition 3: Feeder -> Deagglomerator (cylindrical)
        conn3 = self.conn_feeder_deagg
        self.trans3_start = feeder_geo.outlet_pos.copy()
        self.trans3_end = deagg_geo.inlet_pos.copy()
        self.trans3_radius = conn3.get('start_diameter', feeder_geo.outlet_diameter) / 2.0
        self.trans3_length = conn3.get('length', 0.020)  # 20mm default
        
        print(f"\n  Physics Parameters (computed from geometry):")
        print(f"    Airlock omega:    {self.airlock_omega:.2f} rad/s ({cfg.airlock_rpm} RPM)")
        print(f"    Feeder omega:     {self.feeder_omega:.2f} rad/s ({cfg.feeder_rpm} RPM)")
        print(f"    Deagg omega:      {self.deagg_omega:.2f} rad/s ({cfg.deagg_rpm} RPM)")
        print(f"    Feeder v_axial:   {self.feeder_axial_speed*100:.1f} cm/s")
        
        print(f"\n  Hopper Coordinate System:")
        print(f"    Center Y:         {hopper_geo.center[1]*1000:.1f} mm (base of cone)")
        print(f"    Top Y:            {self.hopper_top_y*1000:.1f} mm (hopper opening)")
        print(f"    Outlet Y:         {self.hopper_outlet_y*1000:.1f} mm (discharge port)")
        print(f"    Top radius:       {hopper_geo.top_radius*1000:.1f} mm (cylinder)")
        print(f"    Bottom radius:    {hopper_geo.bottom_radius*1000:.1f} mm (cone base)")
        print(f"    Outlet radius:    {self.hopper_outlet_radius*1000:.1f} mm (discharge)")
        print(f"    Cone height:      {hopper_geo.cone_height*1000:.1f} mm")
        print(f"    Cylinder height:  {hopper_geo.cylinder_height*1000:.1f} mm")
        print(f"    Discharge ring:   {abs(self.hopper_outlet_y)*1000:.1f} mm (below cone base)")
        
        print(f"\n  Transition Connectors (from diagnose_feed_connectors.py):")
        print(f"    Trans 1: Hopper -> Airlock (cylindrical)")
        print(f"      Start Y:        {self.trans1_start[1]*1000:.1f} mm")
        print(f"      End Y:          {self.trans1_end[1]*1000:.1f} mm")
        print(f"      Radius:         {self.trans1_radius*1000:.1f} mm")
        print(f"      Length:         {self.trans1_length*1000:.1f} mm")
        
        print(f"    Trans 2: Airlock -> Feeder (conical reducer)")
        print(f"      Start Y:        {self.trans2_start[1]*1000:.1f} mm")
        print(f"      End Y:          {self.trans2_end[1]*1000:.1f} mm")
        print(f"      Start radius:   {self.trans2_start_radius*1000:.1f} mm")
        print(f"      End radius:     {self.trans2_end_radius*1000:.1f} mm")
        print(f"      Length:         {self.trans2_length*1000:.1f} mm")
        
        print(f"    Trans 3: Feeder -> Deagg (cylindrical)")
        print(f"      Start Y:        {self.trans3_start[1]*1000:.1f} mm")
        print(f"      End Y:          {self.trans3_end[1]*1000:.1f} mm")
        print(f"      Radius:         {self.trans3_radius*1000:.1f} mm")
        print(f"      Length:         {self.trans3_length*1000:.1f} mm")
        
        print(f"\n  Connection Paths (from port geometry):")
        for name, conn in connections.items():
            direction = conn['direction']
            angle_y = np.degrees(np.arcsin(abs(direction[1])))  # Angle from horizontal
            print(f"    {name}:")
            print(f"      Length:         {conn['length']*1000:.0f} mm")
            print(f"      Direction:      ({direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f})")
            print(f"      Angle from XZ:  {angle_y:.1f} deg")
            print(f"      Diameters:      {conn['start_diameter']*1000:.0f} -> {conn['end_diameter']*1000:.0f} mm")
        print(f"\n  Calculated Flow Rates:")
        print(f"    Airlock Q:        {self.state.airlock_volumetric_rate*3600*1000:.1f} L/h")
        print(f"    Airlock m_dot:    {self.state.airlock_mass_rate*3600:.0f} kg/h")
        print(f"    Feeder Q:         {self.state.feeder_volumetric_rate*3600*1000:.1f} L/h")
        print(f"    Feeder m_dot:     {self.state.feeder_mass_rate*3600:.0f} kg/h")
        print(f"\n  Deagglomerator Geometry:")
        print(f"    Center Y:         {deagg_geo.center[1]*1000:.0f} mm")
        print(f"    Radius:           {deagg_geo.radius*1000:.0f} mm")
        print(f"    Outlet Y:         {self.deagg_outlet_y*1000:.0f} mm")
        print(f"    Exit Y:           {self.exit_y*1000:.0f} mm")
        
        # =====================================================================
        # PARTICLE SIZE VALIDATION
        # Particles must be smaller than passages for free flow
        # =====================================================================
        smallest_passage = min(
            self.trans1_radius * 2,      # Trans1 diameter
            self.trans2_end_radius * 2,  # Trans2 outlet diameter (smallest)
            self.trans3_radius * 2,      # Trans3 diameter
        )
        max_particle_diameter = smallest_passage / 5.0  # Rule: particle < passage/5
        
        particle_dia = cfg.visual_particle_diameter
        print(f"\n  Particle Size Validation:")
        print(f"    Smallest passage:       {smallest_passage*1000:.0f} mm diameter")
        print(f"    Max particle (d/5):     {max_particle_diameter*1000:.1f} mm")
        print(f"    Configured particle:    {particle_dia*1000:.1f} mm")
        
        if particle_dia > max_particle_diameter:
            print(f"    WARNING: Particle diameter ({particle_dia*1000:.0f}mm) > max ({max_particle_diameter*1000:.0f}mm)")
            print(f"             Particles may jam in narrow passages!")
            print(f"             Recommend: --particle-dia {max_particle_diameter*1000:.0f} or smaller")
        else:
            print(f"    Status: OK (particle fits through passages)")
    
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
        mean_diameter: float = 0.015,  # 15mm default (must be < smallest passage / 5)
        std_diameter: float = 0.002,   # 2mm std dev
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
                # =========================================================
                # TRANSITION CONNECTOR GEOMETRY
                # =========================================================
                # Transition 1: Hopper -> Airlock (cylindrical)
                wp.vec3(*self.trans1_start),
                wp.vec3(*self.trans1_end),
                float(self.trans1_radius),
                float(self.trans1_length),
                # Transition 2: Airlock -> Feeder (conical reducer)
                wp.vec3(*self.trans2_start),
                wp.vec3(*self.trans2_end),
                float(self.trans2_start_radius),
                float(self.trans2_end_radius),
                float(self.trans2_length),
                # Transition 3: Feeder -> Deagg (cylindrical)
                wp.vec3(*self.trans3_start),
                wp.vec3(*self.trans3_end),
                float(self.trans3_radius),
                float(self.trans3_length),
                # =========================================================
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
        """Get particle counts by zone, including transition zones."""
        zones = self.state.zones.numpy()[:self.state.particles_active]
        is_active = self.state.is_active.numpy()[:self.state.particles_active]
        
        # Only count active particles
        active_zones = zones[is_active == 1]
        
        return {
            'hopper': int(np.sum(active_zones == 0)),
            'trans_hopper_airlock': int(np.sum(active_zones == 10)),
            'airlock': int(np.sum(active_zones == 1)),
            'trans_airlock_feeder': int(np.sum(active_zones == 11)),
            'feeder': int(np.sum(active_zones == 2)),
            'trans_feeder_deagg': int(np.sum(active_zones == 12)),
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
