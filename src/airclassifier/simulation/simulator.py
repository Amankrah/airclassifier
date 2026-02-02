"""
System-level simulators for air classifier.

Provides separate simulation modes for each subsystem and a complete
integrated simulation:

1. AirSystemSimulator: Flow through blower, filter, dampers
2. FeedSystemSimulator: Material flow through hopper, airlock, feeder, deagglomerator
3. ClassificationSystemSimulator: Particle separation through venturi, zigzag, cyclones, bag filter
4. CompleteSystemSimulator: Full integrated simulation with all systems running

Each simulator can operate independently or be coupled for full system analysis.

Two flow modes are available:
- ANALYTICAL: Fast Rankine vortex / analytical flow models
- CFD: Full Navier-Stokes with CFD-DEM coupling
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
import warp as wp
import yaml

from ..utils.constants import GRAVITY, AirProperties, PI


class FlowMode(Enum):
    """Flow field simulation mode."""
    ANALYTICAL = "analytical"  # Fast analytical flow models
    CFD = "cfd"                # Full Navier-Stokes CFD-DEM coupling


class SystemState(Enum):
    """Operating state of a system."""
    OFF = "off"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"


# =============================================================================
# BASE CONFIGURATION AND STATE
# =============================================================================

@dataclass
class BaseSimulationConfig:
    """Base configuration for all simulators."""
    
    # Flow mode
    flow_mode: FlowMode = FlowMode.ANALYTICAL
    
    # Time parameters
    dt: float = 1.0e-5              # [s] Time step
    duration: float = 1.0           # [s] Total simulation time
    output_interval: float = 0.01   # [s] Output interval
    
    # Physics options
    include_gravity: bool = True
    include_drag: bool = True
    include_wall_collisions: bool = True
    
    # Wall collision parameters
    wall_restitution: float = 0.8
    wall_friction: float = 0.3
    
    # Device
    device: str = "cuda"
    
    # Output
    output_directory: str = "./results"
    
    @property
    def num_steps(self) -> int:
        """Total number of time steps."""
        return int(self.duration / self.dt)
    
    @property
    def output_steps(self) -> int:
        """Steps between outputs."""
        return max(1, int(self.output_interval / self.dt))


@dataclass
class BaseSimulationState:
    """Base state for all simulators."""
    
    # Time
    time: float = 0.0
    step: int = 0
    
    # System state
    system_state: SystemState = SystemState.OFF


# =============================================================================
# AIR SYSTEM SIMULATOR
# =============================================================================

@dataclass
class AirSystemConfig(BaseSimulationConfig):
    """Configuration for air system simulation."""
    
    # Blower parameters
    blower_rpm: float = 3000.0       # [RPM] Blower speed
    blower_ramp_time: float = 0.5    # [s] Time to reach full speed
    
    # Damper settings (0=closed, 1=fully open)
    damper_positions: List[float] = field(default_factory=lambda: [1.0, 1.0])
    
    # Flow parameters
    design_flow_rate_m3_h: float = 3000.0    # [m³/h] Design flow rate
    design_pressure_rise_Pa: float = 5000.0  # [Pa] Design pressure rise


@dataclass
class AirSystemState(BaseSimulationState):
    """State for air system simulation."""
    
    # Blower state
    blower_rpm: float = 0.0
    blower_target_rpm: float = 0.0
    
    # Flow state
    flow_rate_m3_h: float = 0.0
    pressure_Pa: float = 0.0
    
    # Damper states
    damper_positions: List[float] = field(default_factory=lambda: [1.0, 1.0])
    
    # Energy tracking
    power_consumption_kW: float = 0.0
    total_energy_kWh: float = 0.0


class AirSystemSimulator:
    """
    Simulator for the air system (blower + filter + dampers).
    
    Simulates:
    - Blower startup/shutdown ramp
    - Flow rate based on blower speed and damper positions
    - Pressure drop through filter and dampers
    - Power consumption
    
    Flow path:
    Ambient Air -> Filter -> Blower -> Damper(s) -> Outlet
    """
    
    def __init__(
        self,
        assembly: Any,  # AirSystemAssembly
        config: AirSystemConfig = None,
    ):
        """
        Initialize air system simulator.
        
        Args:
            assembly: AirSystemAssembly instance
            config: Simulation configuration
        """
        self.assembly = assembly
        self.config = config or AirSystemConfig()
        self.state = AirSystemState()
        
        # Initialize Warp
        wp.init()
        self.device = self.config.device
        
        # Get system parameters from assembly
        self._setup_system_parameters()
    
    def _setup_system_parameters(self):
        """Extract parameters from assembly."""
        self.blower = self.assembly.blower
        self.dampers = self.assembly.dampers
        self.inlet_filter = self.assembly.inlet_filter
        
        # Blower characteristics
        self.max_rpm = 3600.0  # Typical max
        self.design_rpm = self.config.blower_rpm
        
        # Flow coefficient (simplified fan law)
        self.flow_coefficient = self.config.design_flow_rate_m3_h / self.design_rpm
        
    def start(self):
        """Start the air system (begin blower ramp-up)."""
        self.state.system_state = SystemState.STARTING
        self.state.blower_target_rpm = self.config.blower_rpm
        
    def stop(self):
        """Stop the air system (begin blower ramp-down)."""
        self.state.system_state = SystemState.STOPPING
        self.state.blower_target_rpm = 0.0
        
    def set_damper_position(self, damper_index: int, position: float):
        """Set damper position (0=closed, 1=fully open)."""
        if 0 <= damper_index < len(self.state.damper_positions):
            self.state.damper_positions[damper_index] = max(0.0, min(1.0, position))
    
    def step(self):
        """Advance simulation by one time step."""
        dt = self.config.dt
        
        # Update blower RPM (ramp up/down)
        if self.state.system_state in [SystemState.STARTING, SystemState.RUNNING]:
            ramp_rate = self.design_rpm / self.config.blower_ramp_time
            
            if self.state.blower_rpm < self.state.blower_target_rpm:
                self.state.blower_rpm = min(
                    self.state.blower_rpm + ramp_rate * dt,
                    self.state.blower_target_rpm
                )
            
            if self.state.blower_rpm >= self.state.blower_target_rpm * 0.99:
                self.state.system_state = SystemState.RUNNING
                
        elif self.state.system_state == SystemState.STOPPING:
            ramp_rate = self.design_rpm / self.config.blower_ramp_time
            self.state.blower_rpm = max(
                self.state.blower_rpm - ramp_rate * dt,
                0.0
            )
            
            if self.state.blower_rpm <= 0.01:
                self.state.system_state = SystemState.OFF
                self.state.blower_rpm = 0.0
        
        # Calculate flow rate (simplified fan law: Q ∝ N)
        damper_factor = np.prod(self.state.damper_positions)  # Combined damper effect
        self.state.flow_rate_m3_h = (
            self.flow_coefficient * self.state.blower_rpm * damper_factor
        )
        
        # Calculate pressure (simplified: P ∝ N²)
        rpm_ratio = self.state.blower_rpm / self.design_rpm if self.design_rpm > 0 else 0
        self.state.pressure_Pa = self.config.design_pressure_rise_Pa * rpm_ratio ** 2
        
        # Calculate power (simplified: W ∝ N³)
        design_power_kW = 5.0  # Typical blower power
        self.state.power_consumption_kW = design_power_kW * rpm_ratio ** 3
        self.state.total_energy_kWh += self.state.power_consumption_kW * dt / 3600.0
        
        # Update time
        self.state.time += dt
        self.state.step += 1
    
    def run(self, progress_callback=None):
        """Run the full simulation."""
        total_steps = self.config.num_steps
        
        # Auto-start the system
        self.start()
        
        for step in range(total_steps):
            self.step()
            
            if progress_callback and step % self.config.output_steps == 0:
                progress_callback(step, total_steps)
        
        wp.synchronize()
    
    def get_results(self) -> Dict[str, Any]:
        """Get simulation results."""
        return {
            "time": self.state.time,
            "steps": self.state.step,
            "system_state": self.state.system_state.value,
            "blower_rpm": self.state.blower_rpm,
            "flow_rate_m3_h": self.state.flow_rate_m3_h,
            "pressure_Pa": self.state.pressure_Pa,
            "power_consumption_kW": self.state.power_consumption_kW,
            "total_energy_kWh": self.state.total_energy_kWh,
            "damper_positions": self.state.damper_positions.copy(),
        }
    
    def get_flow_velocity(self, position: np.ndarray) -> np.ndarray:
        """
        Get flow velocity at a position in the air system.
        
        Args:
            position: 3D position [x, y, z]
            
        Returns:
            Velocity vector [vx, vy, vz]
        """
        # Simplified analytical model
        # Assume flow follows the duct centerline
        flow_rate_m3_s = self.state.flow_rate_m3_h / 3600.0
        duct_area = np.pi * (self.assembly._duct_diameter / 2) ** 2
        
        if duct_area > 0 and flow_rate_m3_s > 0:
            velocity_magnitude = flow_rate_m3_s / duct_area
        else:
            velocity_magnitude = 0.0
        
        # Assume flow in +X direction through main duct
        return np.array([velocity_magnitude, 0.0, 0.0])


# =============================================================================
# GRANULAR SIMULATION KERNELS FOR HOPPER FILLING
# =============================================================================

@wp.func
def hopper_sdf(
    pos: wp.vec3,
    hopper_center: wp.vec3,
    top_radius: float,
    bottom_radius: float,
    cylinder_height: float,
    cone_height: float,
) -> float:
    """
    Signed distance function for a conical hopper.
    
    Negative inside, positive outside.
    The hopper consists of:
    - Cylindrical section at top
    - Conical section below
    
    Origin is at center of bottom discharge.
    """
    # Local position relative to hopper center
    px = pos[0] - hopper_center[0]
    py = pos[1] - hopper_center[1]
    pz = pos[2] - hopper_center[2]
    
    # Radial distance in XZ plane
    r = wp.sqrt(px * px + pz * pz)
    
    # Height boundaries
    cone_top_y = cone_height
    cylinder_top_y = cone_height + cylinder_height
    
    # Determine which section we're in
    if py < 0.0:
        # Below hopper - distance to discharge opening
        dist_to_bottom = -py
        dist_to_edge = r - bottom_radius
        if r < bottom_radius:
            return dist_to_bottom  # Inside discharge opening
        return wp.sqrt(dist_to_bottom * dist_to_bottom + dist_to_edge * dist_to_edge)
    
    elif py < cone_height:
        # In conical section
        # Radius at this height (linear interpolation)
        t = py / cone_height
        radius_at_y = bottom_radius + t * (top_radius - bottom_radius)
        
        # SDF for cone (negative inside)
        cone_angle = wp.atan2(top_radius - bottom_radius, cone_height)
        
        if r < radius_at_y:
            # Inside cone - find distance to wall
            # Distance perpendicular to cone surface
            dr = radius_at_y - r
            dist_to_wall = dr * wp.cos(cone_angle)
            return -dist_to_wall  # Negative inside
        else:
            # Outside cone
            dr = r - radius_at_y
            return dr * wp.cos(cone_angle)
    
    elif py < cylinder_top_y:
        # In cylindrical section
        if r < top_radius:
            # Inside cylinder
            return -(top_radius - r)  # Negative inside
        else:
            # Outside cylinder
            return r - top_radius
    
    else:
        # Above hopper
        dist_above = py - cylinder_top_y
        if r < top_radius:
            return dist_above  # Above opening
        # Outside and above
        return wp.sqrt(dist_above * dist_above + (r - top_radius) * (r - top_radius))


@wp.func
def hopper_normal(
    pos: wp.vec3,
    hopper_center: wp.vec3,
    top_radius: float,
    bottom_radius: float,
    cylinder_height: float,
    cone_height: float,
) -> wp.vec3:
    """Get outward normal at position on/near hopper surface."""
    px = pos[0] - hopper_center[0]
    py = pos[1] - hopper_center[1]
    pz = pos[2] - hopper_center[2]
    
    r = wp.sqrt(px * px + pz * pz)
    
    cone_top_y = cone_height
    
    if r < 1.0e-10:
        # On axis - normal points up or down
        if py < cone_height * 0.5:
            return wp.vec3(0.0, -1.0, 0.0)
        return wp.vec3(0.0, 1.0, 0.0)
    
    # Radial unit vector
    r_unit_x = px / r
    r_unit_z = pz / r
    
    if py < 0.0:
        # Below - normal points down
        return wp.vec3(0.0, -1.0, 0.0)
    
    elif py < cone_height:
        # Conical section - normal is tilted
        cone_angle = wp.atan2(top_radius - bottom_radius, cone_height)
        cos_a = wp.cos(cone_angle)
        sin_a = wp.sin(cone_angle)
        
        # Normal = (cos_a * radial, sin_a, 0) in local coords
        return wp.vec3(cos_a * r_unit_x, sin_a, cos_a * r_unit_z)
    
    else:
        # Cylindrical section - normal is purely radial
        return wp.vec3(r_unit_x, 0.0, r_unit_z)


@wp.kernel
def granular_particle_update_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    diameters: wp.array(dtype=float),
    is_active: wp.array(dtype=wp.int32),
    # Hopper parameters
    hopper_center: wp.vec3,
    top_radius: float,
    bottom_radius: float,
    cylinder_height: float,
    cone_height: float,
    discharge_open: int,  # 0 = closed, 1 = open
    # Physics parameters
    dt: float,
    gravity: float,
    restitution: float,
    friction: float,
    damping: float,
):
    """
    Update particles with gravity and hopper collision.
    
    Particles collide with hopper walls and accumulate like granular material.
    """
    tid = wp.tid()
    
    if is_active[tid] == 0:
        return
    
    pos = positions[tid]
    vel = velocities[tid]
    d = diameters[tid]
    radius = d * 0.5
    
    # Apply gravity
    vel = vel + wp.vec3(0.0, -gravity * dt, 0.0)
    
    # Apply damping (air resistance for flour)
    vel = vel * (1.0 - damping * dt)
    
    # Update position
    new_pos = pos + vel * dt
    
    # Local coordinates relative to hopper center (bottom of discharge)
    px = new_pos[0] - hopper_center[0]
    py = new_pos[1] - hopper_center[1]  # Height above discharge
    pz = new_pos[2] - hopper_center[2]
    r = wp.sqrt(px * px + pz * pz)
    
    # Hopper geometry heights
    cone_top_y = cone_height
    cylinder_top_y = cone_height + cylinder_height
    
    # ====== WALL COLLISION ======
    # Apply wall collision for particles inside hopper AND constrain particles above opening
    if py >= 0.0:
        # Calculate wall radius at this height
        if py < cone_height:
            # In cone section - radius varies linearly with height
            t = py / cone_height
            wall_radius = bottom_radius + t * (top_radius - bottom_radius)
        elif py < cylinder_top_y:
            # In cylinder section - constant radius
            wall_radius = top_radius
        else:
            # Above cylinder (above opening) - constrain to opening radius
            # Particles can only be above the opening, not outside it
            wall_radius = top_radius
        
        # Check if particle is hitting or through the wall
        dist_from_wall = wall_radius - r  # Positive when inside, negative when outside
        
        if dist_from_wall < radius:
            # Particle is too close to or through the wall - push inward HARD
            penetration = radius - dist_from_wall
            
            # Radial inward normal
            if r > 1.0e-6:
                nx = -px / r
                nz = -pz / r
            else:
                nx = 0.0
                nz = 0.0
            
            # For cone section, add upward component to normal
            if py < cone_height:
                cone_angle = wp.atan2(top_radius - bottom_radius, cone_height)
                ny = wp.sin(cone_angle)
                horiz = wp.cos(cone_angle)
                normal = wp.normalize(wp.vec3(nx * horiz, ny, nz * horiz))
            else:
                normal = wp.vec3(nx, 0.0, nz)
            
            # Push particle inward with extra margin
            new_pos = new_pos + normal * (penetration + radius * 0.1)
            
            # Strongly reflect velocity off wall
            v_normal = wp.dot(vel, normal)
            if v_normal < 0.0:  # Moving outward (toward wall)
                v_normal_vec = normal * v_normal
                v_tangent = vel - v_normal_vec
                vel = v_tangent * (1.0 - friction) - v_normal_vec * restitution
    
    # ====== ADDITIONAL CONTAINMENT: Force particles inside hopper radius ======
    # This catches any particles that escaped through numerical errors
    px = new_pos[0] - hopper_center[0]
    py = new_pos[1] - hopper_center[1]
    pz = new_pos[2] - hopper_center[2]
    r = wp.sqrt(px * px + pz * pz)
    
    # Calculate max allowed radius at this height
    if py < 0.0:
        max_r = bottom_radius * 2.0  # Below discharge, wider tolerance
    elif py < cone_height:
        t = py / cone_height
        max_r = bottom_radius + t * (top_radius - bottom_radius)
    else:
        max_r = top_radius
    
    # Hard clamp to prevent escapes
    if r > max_r - radius * 0.5:
        if r > 0.01:
            factor = (max_r - radius * 0.6) / r
            new_pos = wp.vec3(hopper_center[0] + px * factor, new_pos[1], hopper_center[2] + pz * factor)
            # Kill outward velocity
            radial_v = (px * vel[0] + pz * vel[2]) / r
            if radial_v > 0.0:
                vel = wp.vec3(vel[0] - px / r * radial_v, vel[1], vel[2] - pz / r * radial_v)
    
    # ====== DISCHARGE BLOCKING ======
    # Recalculate local coords after wall collision
    px = new_pos[0] - hopper_center[0]
    py = new_pos[1] - hopper_center[1]
    pz = new_pos[2] - hopper_center[2]
    r = wp.sqrt(px * px + pz * pz)
    
    if discharge_open == 0:
        # Keep particles above discharge opening
        min_y = radius * 1.5
        if py < min_y and r < bottom_radius + radius:
            new_pos = wp.vec3(new_pos[0], hopper_center[1] + min_y, new_pos[2])
            vel = wp.vec3(vel[0] * 0.3, 0.0, vel[2] * 0.3)
    
    # ====== FLOOR COLLISION (safety) ======
    floor_y = hopper_center[1] - 0.3
    if new_pos[1] < floor_y + radius:
        new_pos = wp.vec3(new_pos[0], floor_y + radius, new_pos[2])
        vel = wp.vec3(0.0, 0.0, 0.0)
    
    positions[tid] = new_pos
    velocities[tid] = vel


@wp.kernel
def particle_particle_collision_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    diameters: wp.array(dtype=float),
    is_active: wp.array(dtype=wp.int32),
    grid: wp.uint64,
    num_particles: int,
    restitution: float,
    friction: float,
    neighbor_radius: float,
):
    """
    Handle particle-particle collisions using hash grid.
    
    Uses DEM-style contact mechanics for granular material.
    """
    tid = wp.tid()
    
    if tid >= num_particles:
        return
    
    if is_active[tid] == 0:
        return
    
    pos_i = positions[tid]
    vel_i = velocities[tid]
    d_i = diameters[tid]
    r_i = d_i * 0.5
    
    # Query neighbors
    query = wp.hash_grid_query(grid, pos_i, neighbor_radius)
    neighbor_idx = int(0)
    
    collision_impulse = wp.vec3(0.0, 0.0, 0.0)
    num_collisions = int(0)
    
    while wp.hash_grid_query_next(query, neighbor_idx):
        if neighbor_idx == tid:
            continue
        
        if is_active[neighbor_idx] == 0:
            continue
        
        pos_j = positions[neighbor_idx]
        d_j = diameters[neighbor_idx]
        r_j = d_j * 0.5
        
        # Vector from j to i
        diff = pos_i - pos_j
        dist = wp.length(diff)
        
        contact_dist = r_i + r_j
        
        if dist < contact_dist and dist > 1.0e-10:
            # Collision detected
            normal = diff / dist
            penetration = contact_dist - dist
            
            # Relative velocity
            vel_j = velocities[neighbor_idx]
            v_rel = vel_i - vel_j
            v_normal = wp.dot(v_rel, normal)
            
            if v_normal < 0.0:
                # Approaching - apply impulse
                # Simple spring-damper model
                impulse_mag = -v_normal * (1.0 + restitution) * 0.5
                impulse = normal * impulse_mag
                
                # Separation force
                separation = normal * penetration * 0.5
                
                collision_impulse = collision_impulse + impulse + separation
                num_collisions = num_collisions + 1
    
    # Apply accumulated impulse
    if num_collisions > 0:
        velocities[tid] = vel_i + collision_impulse


@wp.kernel
def continuous_flow_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    zones: wp.array(dtype=wp.int32),
    is_active: wp.array(dtype=wp.int32),
    num_particles: int,
    # Hopper parameters
    hopper_center: wp.vec3,
    hopper_bottom_radius: float,
    hopper_discharge_y: float,
    # Airlock parameters
    airlock_center: wp.vec3,
    airlock_radius: float,
    airlock_length: float,
    airlock_rpm: float,
    airlock_angle: float,  # Current rotation angle for metering
    # Feeder parameters
    feeder_inlet: wp.vec3,  # Inlet position (under airlock)
    feeder_axis: wp.vec3,   # Feeder direction (normalized)
    feeder_radius: float,
    feeder_length: float,
    feeder_rpm: float,
    feeder_pitch: float,
    feeder_angle: float,    # Current screw rotation angle
    # Deagg parameters  
    deagg_center: wp.vec3,
    deagg_radius: float,
    deagg_length: float,
    deagg_rpm: float,
    deagg_angle: float,     # Current rotor angle
    # Exit
    exit_y: float,
    # Physics
    dt: float,
    gravity: float,
    discharge_open: int,
):
    """
    CONTINUOUS physics-based flow through feed system.
    
    The flow is driven by component rotations:
    - Hopper: Gravity pulls particles to discharge, metered by airlock pocket
    - Airlock: Vanes catch particles, rotate them to outlet - METERS the flow
    - Screw Feeder: Helix pushes particles axially - rate = pitch * RPM
    - Deagglomerator: High-speed pins fling particles through screen
    
    Zone transitions happen naturally based on position, not teleportation.
    """
    tid = wp.tid()
    
    if tid >= num_particles:
        return
    
    if is_active[tid] == 0:
        return
    
    pos = positions[tid]
    vel = velocities[tid]
    zone = zones[tid]
    
    # Angular velocities (rad/s)
    airlock_omega = airlock_rpm * 2.0 * 3.14159 / 60.0
    feeder_omega = feeder_rpm * 2.0 * 3.14159 / 60.0
    deagg_omega = deagg_rpm * 2.0 * 3.14159 / 60.0
    
    # ===== ZONE 0: HOPPER (Gravity-driven, metered by discharge) =====
    if zone == 0:
        # Gravity
        vel = vel + wp.vec3(0.0, -gravity * dt, 0.0)
        
        # Check position relative to hopper
        py = pos[1] - hopper_center[1]
        px = pos[0] - hopper_center[0]
        pz = pos[2] - hopper_center[2]
        r = wp.sqrt(px * px + pz * pz)
        
        # CONTAINMENT: Ensure particles stay within hopper (backup for granular kernel)
        # This prevents escape through numerical errors
        max_r = hopper_bottom_radius * 8.0  # Rough hopper top radius
        if r > max_r * 0.95:
            if r > 0.01:
                factor = max_r * 0.9 / r
                pos = wp.vec3(hopper_center[0] + px * factor, pos[1], hopper_center[2] + pz * factor)
                # Kill outward velocity
                radial_v = (px * vel[0] + pz * vel[2]) / r
                if radial_v > 0.0:
                    vel = wp.vec3(vel[0] - px / r * radial_v * 1.5, vel[1], vel[2] - pz / r * radial_v * 1.5)
        
        if discharge_open == 1:
            # Near discharge opening - funnel toward center
            if py < hopper_bottom_radius * 4.0:
                # Stronger pull toward discharge center
                if r > hopper_bottom_radius * 0.3:
                    pull_strength = 1.0 * dt
                    norm_r = wp.sqrt(px * px + pz * pz)
                    if norm_r > 0.001:
                        vel = vel + wp.vec3(-px / norm_r * pull_strength, 0.0, -pz / norm_r * pull_strength)
            
            # Transition when passing through discharge hole
            if py < -hopper_bottom_radius * 0.2 and r < hopper_bottom_radius * 1.2:
                zones[tid] = 1
                # Place at airlock inlet with downward velocity
                vel = wp.vec3(vel[0] * 0.2, -0.5, vel[2] * 0.2)
    
    # ===== ZONE 1: AIRLOCK (Rotational metering) =====
    # Airlock: horizontal cylinder, rotor rotates around Z-axis
    # - Housing has inlet hole at +Y (top), outlet hole at -Y (bottom)
    # - Particles enter pocket at top, rotor carries them around, exit at bottom
    # - Constrain X position (side walls), allow Y movement (through flow)
    elif zone == 1:
        # Position relative to airlock center
        px = pos[0] - airlock_center[0]
        py = pos[1] - airlock_center[1]
        pz = pos[2] - airlock_center[2]
        
        # Gravity pulls particles down through airlock
        vel = vel + wp.vec3(0.0, -gravity * dt, 0.0)
        
        # Rotational motion - vanes push particles tangentially
        # Distance from rotation axis (Z)
        r_from_axis = wp.sqrt(px * px + py * py)
        
        if r_from_axis > 0.02:
            # Tangent direction around Z axis
            tang_x = -py / r_from_axis
            tang_y = px / r_from_axis
            
            # Vanes carry particles with rotational velocity (scaled by distance from axis)
            v_tangential = airlock_omega * r_from_axis * 0.7
            
            # Apply rotational push
            vel = vel + wp.vec3(tang_x * v_tangential * dt * 10.0, tang_y * v_tangential * dt * 10.0, 0.0)
        
        # Constrain to airlock housing - only in X direction (side walls)
        # Allow free movement in Y for through-flow
        if wp.abs(px) > airlock_radius * 0.85:
            # Push back from side walls
            sign_x = 1.0 if px > 0.0 else -1.0
            pos = wp.vec3(airlock_center[0] + sign_x * airlock_radius * 0.8, pos[1], pos[2])
            if px * vel[0] > 0.0:  # Moving toward wall
                vel = wp.vec3(-vel[0] * 0.3, vel[1], vel[2])
        
        # Constrain top (don't go back up through inlet once inside)
        if py > airlock_radius * 0.9:
            pos = wp.vec3(pos[0], airlock_center[1] + airlock_radius * 0.85, pos[2])
            if vel[1] > 0.0:
                vel = wp.vec3(vel[0], 0.0, vel[2])
        
        # Axial containment along Z (rotor length)
        half_len = airlock_length / 2.0
        if pz < -half_len:
            pos = wp.vec3(pos[0], pos[1], airlock_center[2] - half_len)
        if pz > half_len:
            pos = wp.vec3(pos[0], pos[1], airlock_center[2] + half_len)
        
        # Transition to feeder: particles exit at bottom of airlock (-Y direction)
        if py < -airlock_radius * 0.7:
            zones[tid] = 2
            # Place at feeder inlet with entry velocity
            pos = wp.vec3(feeder_inlet[0], feeder_inlet[1], feeder_inlet[2])
            vel = wp.vec3(feeder_axis[0] * 0.5, -0.2, feeder_axis[2] * 0.5)
    
    # ===== ZONE 2: SCREW FEEDER (Axial push from helix) =====
    elif zone == 2:
        # Calculate conveying velocity from screw
        # V_axial = pitch * RPM / 60 (m/s)
        axial_speed = feeder_pitch * feeder_rpm / 60.0
        
        # Position along feeder axis
        dx = pos[0] - feeder_inlet[0]
        dy = pos[1] - feeder_inlet[1]
        dz = pos[2] - feeder_inlet[2]
        
        # Project onto axis to get axial distance
        axial_dist = dx * feeder_axis[0] + dy * feeder_axis[1] + dz * feeder_axis[2]
        
        # Radial position (perpendicular to axis)
        radial_x = dx - axial_dist * feeder_axis[0]
        radial_y = dy - axial_dist * feeder_axis[1]
        radial_z = dz - axial_dist * feeder_axis[2]
        radial_r = wp.sqrt(radial_x * radial_x + radial_y * radial_y + radial_z * radial_z)
        
        # Screw helix pushes particles axially
        # Also adds some rotational motion (particles tumble)
        tang_scale = 0.3 * feeder_omega * radial_r  # Tangential component
        
        # Main velocity is axial from screw
        vel = wp.vec3(
            feeder_axis[0] * axial_speed + radial_z * tang_scale * 0.1,
            feeder_axis[1] * axial_speed - gravity * dt * 0.3,  # Reduced gravity (supported by trough)
            feeder_axis[2] * axial_speed - radial_x * tang_scale * 0.1
        )
        
        # Constrain to tube
        if radial_r > feeder_radius * 0.85:
            if radial_r > 0.01:
                factor = feeder_radius * 0.8 / radial_r
                pos = wp.vec3(
                    feeder_inlet[0] + axial_dist * feeder_axis[0] + radial_x * factor,
                    feeder_inlet[1] + axial_dist * feeder_axis[1] + radial_y * factor,
                    feeder_inlet[2] + axial_dist * feeder_axis[2] + radial_z * factor
                )
        
        # Transition when reaching feeder outlet
        if axial_dist > feeder_length * 0.9:
            zones[tid] = 3
            # Enter deagglomerator from inlet
            pos = wp.vec3(deagg_center[0] - deagg_length * 0.3, deagg_center[1] + deagg_radius * 0.3, deagg_center[2])
            vel = wp.vec3(0.3, -0.3, 0.0)
    
    # ===== ZONE 3: DEAGGLOMERATOR (High-speed impact + screen discharge) =====
    elif zone == 3:
        # Position relative to deagg center
        px = pos[0] - deagg_center[0]
        py = pos[1] - deagg_center[1]
        pz = pos[2] - deagg_center[2]
        
        # Radial distance in YZ plane (rotor rotates around X)
        r_yz = wp.sqrt(py * py + pz * pz)
        
        # High-speed rotational acceleration from pin impacts
        if r_yz > 0.01:
            # Tangent direction (around X axis)
            tang_y = -pz / r_yz
            tang_z = py / r_yz
            
            # High-speed tangential velocity from pins
            v_tang = deagg_omega * r_yz * 0.6  # Pins impart angular momentum
            
            # Centrifugal force throws particles outward
            centrifugal = deagg_omega * deagg_omega * r_yz * 0.3
            
            vel = wp.vec3(
                vel[0] * 0.95,  # Some axial movement
                tang_y * v_tang + py / r_yz * centrifugal * dt,
                tang_z * v_tang + pz / r_yz * centrifugal * dt
            )
        
        # Gravity
        vel = vel + wp.vec3(0.0, -gravity * dt * 0.5, 0.0)
        
        # Constrain to screen (particles bounce inside until they exit through holes)
        if r_yz > deagg_radius * 0.9:
            if r_yz > 0.01:
                factor = deagg_radius * 0.85 / r_yz
                pos = wp.vec3(pos[0], deagg_center[1] + py * factor, deagg_center[2] + pz * factor)
                # Bounce inward
                radial_vel = (py * vel[1] + pz * vel[2]) / r_yz
                if radial_vel > 0.0:
                    vel = wp.vec3(vel[0], vel[1] - py / r_yz * radial_vel * 1.5, vel[2] - pz / r_yz * radial_vel * 1.5)
        
        # Axial containment
        half_len = deagg_length / 2.0
        if px < -half_len:
            px = -half_len
            pos = wp.vec3(deagg_center[0] + px, pos[1], pos[2])
        if px > half_len:
            px = half_len
            pos = wp.vec3(deagg_center[0] + px, pos[1], pos[2])
        
        # Exit through bottom (screen discharge)
        if py < -deagg_radius * 0.7:
            zones[tid] = 4
            # Exit with angular momentum
            vel = wp.vec3(vel[0] * 0.3, -1.0, vel[2] * 0.5)
    
    # ===== ZONE 4: EXITED (Free fall after system) =====
    elif zone == 4:
        # Free fall
        vel = vel + wp.vec3(0.0, -gravity * dt, 0.0)
        
        # Air drag
        vel = vel * 0.995
        
        # Floor collision
        if pos[1] < exit_y:
            pos = wp.vec3(pos[0], exit_y, pos[2])
            vel = wp.vec3(vel[0] * 0.3, 0.0, vel[2] * 0.3)
            if wp.abs(vel[0]) < 0.01 and wp.abs(vel[2]) < 0.01:
                is_active[tid] = 0  # Deactivate when settled
    
    # Update position
    pos = pos + vel * dt
    
    positions[tid] = pos
    velocities[tid] = vel


# =============================================================================
# FEED SYSTEM SIMULATOR
# =============================================================================

@dataclass
class FeedSystemConfig(BaseSimulationConfig):
    """Configuration for feed system simulation."""
    
    # Feed parameters
    feed_rate_kg_h: float = 500.0    # [kg/h] Target feed rate
    material_bulk_density: float = 500.0  # [kg/m³] Material bulk density
    
    # Component speeds
    airlock_rpm: float = 20.0        # [RPM] Rotary airlock speed
    feeder_rpm: float = 60.0         # [RPM] Screw feeder speed
    deagg_rpm: float = 1500.0        # [RPM] Deagglomerator speed
    
    # Ramp times
    ramp_time: float = 2.0           # [s] Time to reach full speed
    
    # Particle tracking (for non-pouring mode)
    num_particles: int = 5000        # Number of particles to track
    injection_duration: float = 0.5  # [s] Duration over which to inject
    
    # Hopper lid animation
    animate_lid: bool = True         # Whether to animate lid opening/closing
    lid_open_angle: float = 90.0     # [degrees] Maximum lid opening angle
    lid_animation_time: float = 1.0  # [s] Time to open/close lid
    
    # Hopper filling simulation
    enable_pouring: bool = True             # Simulate pouring particles into hopper
    hopper_fill_percentage: float = 50.0    # [%] Percentage of hopper capacity to fill (0-100)
    pour_height: float = 0.3                # [m] Height above hopper to pour from
    pour_rate_kg_s: float = 100.0           # [kg/s] Mass flow rate during pouring
    pour_stream_radius: float = 0.15        # [m] Radius of pour stream
    visual_particle_diameter: float = 0.04  # [m] Display size of visual particles (40mm)
    settling_time: float = 0.5              # [s] Time to wait for particles to settle after pouring
    
    # Particle scaling (parcel method)
    max_visual_particles: int = 1500        # Max particles to simulate (each represents a parcel)
    
    # Granular physics
    enable_particle_collisions: bool = True  # Enable particle-particle collisions
    granular_restitution: float = 0.1        # Low restitution for flour
    granular_friction: float = 0.4           # Friction coefficient
    granular_damping: float = 5.0            # Air damping for flour particles
    hash_grid_dim: int = 64                  # Hash grid resolution
    neighbor_search_radius: float = 0.02     # [m] Neighbor search radius


class LidState(Enum):
    """State of the hopper lid."""
    CLOSED = "closed"
    OPENING = "opening"
    OPEN = "open"
    CLOSING = "closing"


class PouringState(Enum):
    """State of particle pouring."""
    IDLE = "idle"
    POURING = "pouring"
    SETTLING = "settling"  # Waiting for particles to enter hopper
    COMPLETED = "completed"


@dataclass
class FeedSystemState(BaseSimulationState):
    """State for feed system simulation."""
    
    # Component speeds
    airlock_rpm: float = 0.0
    feeder_rpm: float = 0.0
    deagg_rpm: float = 0.0
    
    # Material flow
    mass_flow_rate_kg_h: float = 0.0
    hopper_mass_kg: float = 0.0
    target_fill_mass_kg: float = 0.0    # Target mass to pour
    
    # Particle tracking (on device)
    positions: Optional[wp.array] = None
    velocities: Optional[wp.array] = None
    diameters: Optional[wp.array] = None
    is_active: Optional[wp.array] = None
    
    # Statistics
    particles_injected: int = 0
    particles_discharged: int = 0
    total_particles_to_pour: int = 0    # Calculated from volume/mass
    particles_inside_hopper: int = 0    # Particles that have entered hopper
    
    # Hopper lid animation state
    lid_state: LidState = LidState.CLOSED
    lid_angle: float = 0.0              # [degrees] Current lid angle (0=closed, 90=fully open)
    lid_target_angle: float = 0.0       # [degrees] Target lid angle
    lid_angular_velocity: float = 0.0   # [degrees/s] Current angular velocity
    
    # Pouring state
    pouring_state: PouringState = PouringState.IDLE
    pour_start_time: float = 0.0
    settling_start_time: float = 0.0
    particles_poured: int = 0


class FeedSystemSimulator:
    """
    Simulator for the feed system (hopper + airlock + feeder + deagglomerator).
    
    Simulates:
    - Hopper lid opening/closing animation
    - Particle pouring into hopper from above
    - Material discharge from hopper (gravity flow)
    - Rotary airlock volumetric metering
    - Screw feeder controlled dosing
    - Deagglomerator lump breaking
    
    Flow path:
    [Pour from above] -> Hopper (lid opens) -> Airlock -> Screw Feeder -> Deagglomerator -> Outlet
    """
    
    def __init__(
        self,
        assembly: Any,  # FeedSystemAssembly
        config: FeedSystemConfig = None,
        material: Any = None,  # Optional ParticleMaterial
    ):
        """
        Initialize feed system simulator.
        
        Args:
            assembly: FeedSystemAssembly instance
            config: Simulation configuration
            material: Optional ParticleMaterial for proper size distribution
        """
        self.assembly = assembly
        self.config = config or FeedSystemConfig()
        self.material = material
        self.state = FeedSystemState()
        
        # Initialize Warp
        wp.init()
        self.device = self.config.device
        
        # Calculate fill parameters if pouring is enabled
        if self.config.enable_pouring:
            self._calculate_fill_parameters()
        
        # Initialize hopper mass (starts empty for filling simulation)
        self.state.hopper_mass_kg = 0.0 if self.config.enable_pouring else self.assembly.params.hopper_capacity_kg
        
        # Setup system
        self._setup_system_parameters()
        self._allocate_arrays()
        
        # Lid animation parameters
        self._lid_max_angular_velocity = self.config.lid_open_angle / self.config.lid_animation_time
    
    def _calculate_fill_parameters(self):
        """
        Calculate fill parameters using VOLUME-BASED PARTICLE SIZING.
        
        Calculate the particle diameter needed so that a fixed number of particles
        will visually fill the hopper to the specified percentage.
        
        Key formula:
            particle_volume = (fill_volume * packing_efficiency) / num_particles
            particle_diameter = 2 * (3 * particle_volume / (4 * π))^(1/3)
        """
        # Get hopper geometry
        hopper_capacity_kg = self.assembly.params.hopper_capacity_kg
        hopper = self.assembly.hopper
        
        # Total hopper volume from geometry (not from capacity/density)
        hopper_total_volume_m3 = hopper.params.total_volume
        
        # Calculate target fill
        fill_fraction = self.config.hopper_fill_percentage / 100.0
        target_mass_kg = hopper_capacity_kg * fill_fraction
        fill_volume_m3 = hopper_total_volume_m3 * fill_fraction
        self.state.target_fill_mass_kg = target_mass_kg
        
        # Fixed number of particles
        num_particles = self.config.max_visual_particles
        self.config.num_particles = num_particles
        self.state.total_particles_to_pour = num_particles
        
        # Calculate particle diameter to fill the volume
        # With random packing efficiency (~64%), particles occupy 64% of volume
        packing_efficiency = 0.64
        volume_per_particle = (fill_volume_m3 * packing_efficiency) / num_particles
        
        # Sphere volume: V = (4/3) * π * r³  =>  r = (3V / 4π)^(1/3)
        particle_radius = (3.0 * volume_per_particle / (4.0 * np.pi)) ** (1.0/3.0)
        calculated_diameter = 2.0 * particle_radius
        
        # Store the calculated diameter (override config)
        self._visual_particle_diameter = calculated_diameter
        self.config.visual_particle_diameter = calculated_diameter
        
        # Update neighbor search radius to match particle size
        self.config.neighbor_search_radius = calculated_diameter * 1.1
        
        # Each visual particle represents this much mass
        self._mass_per_parcel = target_mass_kg / num_particles
        
        # Calculate pour duration from mass flow rate
        if self.config.pour_rate_kg_s > 0:
            pour_duration = target_mass_kg / self.config.pour_rate_kg_s
            self._calculated_pour_duration = max(0.3, pour_duration)
        else:
            self._calculated_pour_duration = 1.0
        
        print(f"\n  Fill Calculation (Volume-Based Sizing):")
        print(f"    Hopper volume:       {hopper_total_volume_m3 * 1000:.0f} liters")
        print(f"    Hopper capacity:     {hopper_capacity_kg:.0f} kg")
        print(f"    Fill percentage:     {self.config.hopper_fill_percentage:.0f}%")
        print(f"    Target fill mass:    {target_mass_kg:.1f} kg")
        print(f"    Fill volume:         {fill_volume_m3 * 1000:.0f} liters")
        print(f"    Particles:           {num_particles:,}")
        print(f"    Particle diameter:   {calculated_diameter * 1000:.1f} mm (calculated)")
        print(f"    Mass per particle:   {self._mass_per_parcel:.3f} kg")
        print(f"    Pour duration:       {self._calculated_pour_duration:.1f} s")
    
    def _setup_system_parameters(self):
        """Extract parameters from assembly."""
        self.hopper = self.assembly.hopper
        self.airlock = self.assembly.airlock
        self.feeder = self.assembly.feeder
        self.deagglomerator = self.assembly.deagglomerator
        
        # Get component positions
        self.component_positions = self.assembly.get_component_positions()
        
        # Calculate volumetric flow from airlock
        self.airlock_pocket_volume = (
            np.pi * (self.airlock.params.rotor_diameter / 2) ** 2 *
            self.airlock.params.rotor_length / self.airlock.params.num_vanes
        )
        
        # Get hopper dimensions for pouring and collision
        hopper_pos = np.array(self.component_positions['hopper'])
        self.hopper_center = hopper_pos.copy()  # Bottom center of hopper
        
        self.hopper_top_y = (
            self.component_positions['hopper'][1] + 
            self.hopper.params.total_height
        )
        self.hopper_top_radius = self.hopper.params.top_radius
        self.hopper_bottom_radius = self.hopper.params.bottom_radius
        self.hopper_cylinder_height = self.hopper.params.cylindrical_height
        self.hopper_cone_height = self.hopper.params.conical_height
        
        # Hinge position for lid animation (on -X side of hopper)
        self.lid_hinge_position = np.array([
            hopper_pos[0] - self.hopper_top_radius * 1.08,
            self.hopper_top_y,
            hopper_pos[2]
        ])
        
        # Discharge state (0 = closed during filling, 1 = open for discharge)
        self._discharge_open = 0
        
        # Setup flow path parameters for particle flow simulation
        self._setup_flow_path()
    
    def _setup_flow_path(self):
        """
        Setup flow path parameters for particle simulation through all components.
        
        Flow path: Hopper -> Airlock -> Screw Feeder -> Deagglomerator -> Exit
        """
        # Get component positions
        hopper_pos = np.array(self.component_positions['hopper'])
        airlock_pos = np.array(self.component_positions['airlock'])
        feeder_pos = np.array(self.component_positions['feeder'])
        deagg_pos = np.array(self.component_positions['deagglomerator'])
        
        # Airlock parameters
        airlock = self.assembly.airlock
        self.airlock_center = airlock_pos.copy()
        self.airlock_radius = airlock.params.rotor_diameter / 2
        self.airlock_length = airlock.params.rotor_length
        self.airlock_inlet_y = self.airlock_center[1] + self.airlock_radius
        self.airlock_outlet_y = self.airlock_center[1] - self.airlock_radius
        
        # Screw feeder parameters  
        feeder = self.assembly.feeder
        self.feeder_center = feeder_pos.copy()
        self.feeder_radius = feeder.params.screw_diameter / 2 + feeder.params.trough_clearance
        self.feeder_length = feeder.params.trough_length
        self.feeder_inlet_pos = self.feeder_center.copy()
        self.feeder_outlet_pos = self.feeder_center.copy()
        self.feeder_outlet_pos[0] += self.feeder_length  # Feeder moves along X
        
        # Deagglomerator parameters
        deagg = self.assembly.deagglomerator
        self.deagg_center = deagg_pos.copy()
        self.deagg_radius = deagg.params.housing_diameter / 2
        self.deagg_length = deagg.params.housing_length
        self.deagg_outlet_y = self.deagg_center[1] - self.deagg_radius
        
        # Exit position (below deagglomerator)
        self.exit_y = self.deagg_outlet_y - 0.2
        
        # Flow zone boundaries (Y coordinates)
        self.flow_zones = {
            'hopper': (self.hopper_center[1], self.hopper_top_y),
            'airlock': (self.airlock_outlet_y, self.airlock_inlet_y),
            'feeder': (self.feeder_center[1] - self.feeder_radius, 
                      self.feeder_center[1] + self.feeder_radius),
            'deagg': (self.deagg_outlet_y, self.deagg_center[1] + self.deagg_radius),
            'exit': (self.exit_y - 0.5, self.exit_y),
        }
    
    def _allocate_arrays(self):
        """Pre-allocate particle arrays and hash grid."""
        n = self.config.num_particles
        
        self.state.positions = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.state.velocities = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.state.diameters = wp.zeros(n, dtype=float, device=self.device)
        self.state.is_active = wp.zeros(n, dtype=wp.int32, device=self.device)
        
        # Zone tracking: 0=hopper, 1=airlock, 2=feeder, 3=deagg, 4=exit
        self._particle_zones = wp.zeros(n, dtype=wp.int32, device=self.device)
        
        # Temporary arrays
        self._accelerations = wp.zeros(n, dtype=wp.vec3, device=self.device)
        
        # Hash grid for particle-particle collisions
        if self.config.enable_particle_collisions:
            grid_dim = self.config.hash_grid_dim
            self._hash_grid = wp.HashGrid(
                dim_x=grid_dim, 
                dim_y=grid_dim, 
                dim_z=grid_dim, 
                device=self.device
            )
        
        # Counters for particles in each zone
        self._zone_counts = {'hopper': 0, 'airlock': 0, 'feeder': 0, 'deagg': 0, 'exit': 0}
    
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
    
    def start_pouring(self):
        """Start pouring particles into the hopper."""
        if self.state.pouring_state == PouringState.IDLE:
            # Must open lid first
            self.open_lid()
            self.state.pouring_state = PouringState.POURING
            self.state.pour_start_time = self.state.time
    
    def stop_pouring(self):
        """Stop pouring and close the lid."""
        if self.state.pouring_state == PouringState.POURING:
            self.state.pouring_state = PouringState.COMPLETED
            self.close_lid()
    
    def start(self):
        """Start the feed system."""
        self.state.system_state = SystemState.STARTING
        
        # If pouring is enabled, start the pouring sequence
        if self.config.enable_pouring:
            self.start_pouring()
    
    def stop(self):
        """Stop the feed system."""
        self.state.system_state = SystemState.STOPPING
    
    def _count_particles_inside_hopper(self):
        """
        Count how many particles are currently inside the hopper volume.
        
        Particles are considered "inside" if they are within the hopper geometry
        (below the top opening and above the discharge).
        """
        if self.state.particles_poured == 0:
            self.state.particles_inside_hopper = 0
            return
        
        # Get positions from GPU
        positions = self.state.positions.numpy()[:self.state.particles_poured]
        
        # Hopper boundaries
        hopper_bottom_y = self.hopper_center[1]
        hopper_top_y = hopper_bottom_y + self.hopper_cone_height + self.hopper_cylinder_height
        hopper_cx = self.hopper_center[0]
        hopper_cz = self.hopper_center[2]
        
        count_inside = 0
        for i in range(len(positions)):
            px, py, pz = positions[i]
            
            # Check if below top opening
            if py > hopper_top_y:
                continue
            
            # Check if above discharge
            if py < hopper_bottom_y:
                continue
            
            # Check radial distance at this height
            if py < hopper_bottom_y + self.hopper_cone_height:
                # In cone section - radius varies with height
                t = (py - hopper_bottom_y) / self.hopper_cone_height
                radius_at_y = self.hopper_bottom_radius + t * (self.hopper_top_radius - self.hopper_bottom_radius)
            else:
                # In cylinder section
                radius_at_y = self.hopper_top_radius
            
            # Radial distance
            r = np.sqrt((px - hopper_cx)**2 + (pz - hopper_cz)**2)
            
            if r < radius_at_y:
                count_inside += 1
        
        self.state.particles_inside_hopper = count_inside
    
    def _get_average_particle_velocity(self) -> float:
        """
        Get the average velocity magnitude of all poured particles.
        
        Used to determine when particles have settled.
        """
        if self.state.particles_poured == 0:
            return 0.0
        
        velocities = self.state.velocities.numpy()[:self.state.particles_poured]
        speeds = np.linalg.norm(velocities, axis=1)
        return float(np.mean(speeds))
    
    def _update_zone_counts(self):
        """
        Update counts of particles in each flow zone.
        """
        if not hasattr(self, '_particle_zones'):
            return
        
        zones = self._particle_zones.numpy()[:self.state.particles_poured]
        
        self._zone_counts = {
            'hopper': int(np.sum(zones == 0)),
            'airlock': int(np.sum(zones == 1)),
            'feeder': int(np.sum(zones == 2)),
            'deagg': int(np.sum(zones == 3)),
            'exit': int(np.sum(zones == 4)),
        }
        
        # Update hopper mass based on particles in hopper zone
        if self.state.total_particles_to_pour > 0:
            mass_per_particle = self.state.target_fill_mass_kg / self.state.total_particles_to_pour
            self.state.hopper_mass_kg = self._zone_counts['hopper'] * mass_per_particle
    
    def _update_lid_animation(self, dt: float):
        """Update lid opening/closing animation."""
        if not self.config.animate_lid:
            return
        
        angle_diff = self.state.lid_target_angle - self.state.lid_angle
        
        if abs(angle_diff) < 0.1:
            # Reached target
            self.state.lid_angle = self.state.lid_target_angle
            self.state.lid_angular_velocity = 0.0
            
            if self.state.lid_state == LidState.OPENING:
                self.state.lid_state = LidState.OPEN
            elif self.state.lid_state == LidState.CLOSING:
                self.state.lid_state = LidState.CLOSED
        else:
            # Animate with smooth acceleration/deceleration
            direction = 1.0 if angle_diff > 0 else -1.0
            
            # Simple smooth interpolation
            angular_speed = self._lid_max_angular_velocity
            delta_angle = direction * angular_speed * dt
            
            # Clamp to not overshoot
            if abs(delta_angle) > abs(angle_diff):
                delta_angle = angle_diff
            
            self.state.lid_angle += delta_angle
            self.state.lid_angular_velocity = delta_angle / dt if dt > 0 else 0.0
    
    def get_lid_transform(self) -> np.ndarray:
        """
        Get the 4x4 transformation matrix for the lid.
        
        The lid rotates around the hinge axis (Z-axis at hinge position).
        
        Returns:
            4x4 transformation matrix
        """
        angle_rad = np.radians(self.state.lid_angle)
        
        # Rotation around Z axis at hinge position
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Translation to hinge, rotate, translate back
        hx, hy, hz = self.lid_hinge_position
        
        # Combined transform: T(hinge) * Rz(angle) * T(-hinge)
        transform = np.array([
            [cos_a, -sin_a, 0, hx - hx*cos_a + hy*sin_a],
            [sin_a,  cos_a, 0, hy - hx*sin_a - hy*cos_a],
            [0,      0,     1, 0],
            [0,      0,     0, 1]
        ], dtype=np.float32)
        
        return transform
    
    def pour_particles(self, n_pour: int):
        """
        Pour particles from above into the open hopper.
        
        Creates a stream of particles falling from pour_height above the hopper.
        """
        if self.state.particles_poured >= self.config.num_particles:
            return
        
        n_pour = min(n_pour, self.config.num_particles - self.state.particles_poured)
        
        # Only pour if lid is sufficiently open
        if self.state.lid_angle < self.config.lid_open_angle * 0.5:
            return
        
        rng = np.random.default_rng(self.state.step + 1234)
        
        # Use the calculated visual particle diameter (not microscopic flour diameter)
        # Each visual particle represents a "parcel" of material
        visual_diameter = self.config.visual_particle_diameter
        
        # Small random variation in visual particle size (±10%)
        diameters = np.full(n_pour, visual_diameter, dtype=np.float32)
        diameters *= (1.0 + rng.uniform(-0.1, 0.1, n_pour).astype(np.float32))
        
        # Pour position: above hopper center
        hopper_pos = np.array(self.component_positions['hopper'])
        pour_y = self.hopper_top_y + self.config.pour_height
        
        # Generate positions in a circular stream
        positions = np.zeros((n_pour, 3), dtype=np.float32)
        velocities = np.zeros((n_pour, 3), dtype=np.float32)
        
        # Limit pour stream to fit within hopper opening (with margin)
        max_stream_radius = self.hopper_top_radius * 0.7  # 70% of hopper opening
        stream_radius = min(self.config.pour_stream_radius, max_stream_radius)
        
        for i in range(n_pour):
            # Random position within pour stream (uniform distribution in circle)
            r = np.sqrt(rng.random()) * stream_radius  # sqrt for uniform area distribution
            theta = rng.random() * 2 * np.pi
            
            positions[i, 0] = hopper_pos[0] + r * np.cos(theta)
            positions[i, 1] = pour_y + rng.uniform(-0.01, 0.01)
            positions[i, 2] = hopper_pos[2] + r * np.sin(theta)
            
            # Downward velocity (gravity-driven pour) - more vertical
            velocities[i, 0] = rng.uniform(-0.05, 0.05)  # Less horizontal spread
            velocities[i, 1] = -1.0 - rng.random() * 0.5  # Faster downward
            velocities[i, 2] = rng.uniform(-0.05, 0.05)
        
        # Copy to device
        start_idx = self.state.particles_poured
        
        positions_wp = wp.array(positions, dtype=wp.vec3, device=self.device)
        velocities_wp = wp.array(velocities, dtype=wp.vec3, device=self.device)
        diameters_wp = wp.array(diameters.astype(np.float32), dtype=float, device=self.device)
        
        wp.copy(self.state.positions, positions_wp, dest_offset=start_idx, count=n_pour)
        wp.copy(self.state.velocities, velocities_wp, dest_offset=start_idx, count=n_pour)
        wp.copy(self.state.diameters, diameters_wp, dest_offset=start_idx, count=n_pour)
        
        active_flags = wp.array(np.ones(n_pour, dtype=np.int32), dtype=wp.int32, device=self.device)
        wp.copy(self.state.is_active, active_flags, dest_offset=start_idx, count=n_pour)
        
        self.state.particles_poured += n_pour
        self.state.particles_injected += n_pour
    
    def inject_particles(self, n_inject: int):
        """Inject particles at hopper discharge (when system is running)."""
        if self.state.particles_injected >= self.config.num_particles:
            return
        
        n_inject = min(n_inject, self.config.num_particles - self.state.particles_injected)
        
        # Generate random particle sizes (log-normal distribution)
        rng = np.random.default_rng(self.state.step + 42)
        
        if self.material is not None:
            diameters = self.material.sample_diameters(n_inject, seed=self.state.step)
        else:
            mean_diameter = 50e-6  # 50 microns
            diameters = rng.lognormal(
                mean=np.log(mean_diameter),
                sigma=0.5,
                size=n_inject
            ).astype(np.float32)
            diameters = np.clip(diameters, 5e-6, 500e-6)
        
        # Get hopper discharge position
        hopper_pos = np.array(self.component_positions['hopper'])
        discharge_port = self.hopper.ports['discharge']
        discharge_pos = hopper_pos + np.array(discharge_port.position)
        
        # Generate random positions within discharge area
        positions = np.zeros((n_inject, 3), dtype=np.float32)
        velocities = np.zeros((n_inject, 3), dtype=np.float32)
        
        discharge_radius = discharge_port.diameter / 2 * 0.8
        
        for i in range(n_inject):
            r = rng.random() * discharge_radius
            theta = rng.random() * 2 * np.pi
            
            positions[i, 0] = discharge_pos[0] + r * np.cos(theta)
            positions[i, 1] = discharge_pos[1]
            positions[i, 2] = discharge_pos[2] + r * np.sin(theta)
            
            # Initial velocity (small downward)
            velocities[i, 1] = -0.1
        
        # Copy to device
        start_idx = self.state.particles_injected
        
        positions_wp = wp.array(positions, dtype=wp.vec3, device=self.device)
        velocities_wp = wp.array(velocities, dtype=wp.vec3, device=self.device)
        diameters_wp = wp.array(diameters.astype(np.float32), dtype=float, device=self.device)
        
        wp.copy(self.state.positions, positions_wp, dest_offset=start_idx, count=n_inject)
        wp.copy(self.state.velocities, velocities_wp, dest_offset=start_idx, count=n_inject)
        wp.copy(self.state.diameters, diameters_wp, dest_offset=start_idx, count=n_inject)
        
        active_flags = wp.array(np.ones(n_inject, dtype=np.int32), dtype=wp.int32, device=self.device)
        wp.copy(self.state.is_active, active_flags, dest_offset=start_idx, count=n_inject)
        
        self.state.particles_injected += n_inject
    
    def step(self):
        """Advance simulation by one time step."""
        dt = self.config.dt
        
        # ===== LID ANIMATION =====
        self._update_lid_animation(dt)
        
        # ===== POURING PHASE =====
        if self.state.pouring_state == PouringState.POURING:
            elapsed_pour_time = self.state.time - self.state.pour_start_time
            pour_duration = getattr(self, '_calculated_pour_duration', 2.0)
            
            # Check if we've poured all particles
            if self.state.particles_poured < self.state.total_particles_to_pour:
                # Calculate pour rate based on calculated duration
                pour_rate = self.state.total_particles_to_pour / pour_duration
                n_pour = int(pour_rate * dt) + (1 if np.random.random() < (pour_rate * dt % 1) else 0)
                n_pour = min(n_pour, self.state.total_particles_to_pour - self.state.particles_poured)
                
                if n_pour > 0:
                    self.pour_particles(n_pour)
            else:
                # All particles poured, transition to settling
                self.state.pouring_state = PouringState.SETTLING
                self.state.settling_start_time = self.state.time
        
        # ===== SETTLING PHASE - Wait for particles to settle inside hopper =====
        elif self.state.pouring_state == PouringState.SETTLING:
            settling_elapsed = self.state.time - self.state.settling_start_time
            
            # Check settling state every 20 steps
            if self.state.step % 20 == 0:
                self._count_particles_inside_hopper()
                avg_velocity = self._get_average_particle_velocity()
                
                # Update mass progressively during settling
                if self.state.total_particles_to_pour > 0:
                    mass_per_particle = self.state.target_fill_mass_kg / self.state.total_particles_to_pour
                    self.state.hopper_mass_kg = self.state.particles_inside_hopper * mass_per_particle
                
                # Settling is complete when:
                # 1. ALL particles are inside hopper (100%)
                # 2. Average velocity is low (particles have settled)
                # 3. Minimum settling time elapsed (give particles time to fall)
                particles_inside_ratio = self.state.particles_inside_hopper / max(1, self.state.particles_poured)
                velocity_settled = avg_velocity < 0.10  # Less than 10 cm/s average
                min_time_elapsed = settling_elapsed >= 0.5  # At least 0.5s
                max_time_elapsed = settling_elapsed >= self.config.settling_time * 5  # Safety timeout (longer)
                
                # Close lid when ALL particles settled OR timeout
                all_inside = particles_inside_ratio >= 0.99  # 99%+ inside (account for counting errors)
                if (all_inside and velocity_settled and min_time_elapsed) or max_time_elapsed:
                    # Final count
                    self._count_particles_inside_hopper()
                    if self.state.total_particles_to_pour > 0:
                        mass_per_particle = self.state.target_fill_mass_kg / self.state.total_particles_to_pour
                        self.state.hopper_mass_kg = self.state.particles_inside_hopper * mass_per_particle
                    
                    # Now close the lid
                    self.close_lid()
                    self.state.pouring_state = PouringState.COMPLETED
        
        # ===== COMPONENT SPEED RAMP =====
        if self.state.system_state == SystemState.STARTING:
            # Wait for pouring AND settling to complete before starting components
            if self.state.pouring_state == PouringState.COMPLETED:
                ramp_rate = 1.0 / self.config.ramp_time
                
                self.state.airlock_rpm = min(
                    self.state.airlock_rpm + self.config.airlock_rpm * ramp_rate * dt,
                    self.config.airlock_rpm
                )
                self.state.feeder_rpm = min(
                    self.state.feeder_rpm + self.config.feeder_rpm * ramp_rate * dt,
                    self.config.feeder_rpm
                )
                self.state.deagg_rpm = min(
                    self.state.deagg_rpm + self.config.deagg_rpm * ramp_rate * dt,
                    self.config.deagg_rpm
                )
                
                if (self.state.airlock_rpm >= self.config.airlock_rpm * 0.99 and
                    self.state.feeder_rpm >= self.config.feeder_rpm * 0.99):
                    self.state.system_state = SystemState.RUNNING
                
        elif self.state.system_state == SystemState.STOPPING:
            ramp_rate = 1.0 / self.config.ramp_time
            
            self.state.airlock_rpm = max(0, self.state.airlock_rpm - self.config.airlock_rpm * ramp_rate * dt)
            self.state.feeder_rpm = max(0, self.state.feeder_rpm - self.config.feeder_rpm * ramp_rate * dt)
            self.state.deagg_rpm = max(0, self.state.deagg_rpm - self.config.deagg_rpm * ramp_rate * dt)
            
            if self.state.airlock_rpm <= 0.01:
                self.state.system_state = SystemState.OFF
        
        # ===== MASS FLOW CALCULATION =====
        # Discharge only opens when:
        # 1. System is RUNNING
        # 2. Lid is fully CLOSED
        # 3. Pouring is COMPLETED
        lid_closed = self.state.lid_state == LidState.CLOSED and self.state.lid_angle < 1.0
        pouring_done = self.state.pouring_state == PouringState.COMPLETED
        
        if self.state.system_state == SystemState.RUNNING and lid_closed and pouring_done:
            # Enable discharge when system is running and lid is closed
            self._discharge_open = 1
            
            # Airlock volumetric flow (metered discharge rate)
            vol_flow_m3_s = self.airlock_pocket_volume * (self.state.airlock_rpm / 60.0)
            self.state.mass_flow_rate_kg_h = (
                vol_flow_m3_s * self.config.material_bulk_density * 3600.0
            )
        else:
            self.state.mass_flow_rate_kg_h = 0.0
            self._discharge_open = 0  # Keep discharge closed when not ready
        
        # ===== DISCHARGE INJECTION (when running) =====
        if self.state.system_state == SystemState.RUNNING and not self.config.enable_pouring:
            if self.state.time < self.config.injection_duration:
                inject_rate = self.config.num_particles / self.config.injection_duration
                n_inject = int(inject_rate * dt) + (1 if np.random.random() < (inject_rate * dt % 1) else 0)
                if n_inject > 0:
                    self.inject_particles(n_inject)
        
        # ===== GRANULAR PARTICLE PHYSICS UPDATE =====
        n = self.state.particles_injected
        if n > 0:
            # Hopper center as vec3
            hopper_center = wp.vec3(
                float(self.hopper_center[0]),
                float(self.hopper_center[1]),
                float(self.hopper_center[2])
            )
            
            # Update particles with hopper collision using SDF
            wp.launch(
                kernel=granular_particle_update_kernel,
                dim=n,
                inputs=[
                    self.state.positions,
                    self.state.velocities,
                    self.state.diameters,
                    self.state.is_active,
                    hopper_center,
                    float(self.hopper_top_radius),
                    float(self.hopper_bottom_radius),
                    float(self.hopper_cylinder_height),
                    float(self.hopper_cone_height),
                    self._discharge_open,
                    dt,
                    float(GRAVITY),
                    float(self.config.granular_restitution),
                    float(self.config.granular_friction),
                    float(self.config.granular_damping),
                ],
                device=self.device
            )
            
            # Particle-particle collisions using hash grid
            if self.config.enable_particle_collisions and n > 1:
                # Rebuild hash grid with current particle positions
                self._hash_grid.build(
                    points=self.state.positions,
                    radius=self.config.neighbor_search_radius
                )
                
                # Apply particle-particle collision forces
                wp.launch(
                    kernel=particle_particle_collision_kernel,
                    dim=n,
                    inputs=[
                        self.state.positions,
                        self.state.velocities,
                        self.state.diameters,
                        self.state.is_active,
                        self._hash_grid.id,
                        n,
                        float(self.config.granular_restitution),
                        float(self.config.granular_friction),
                        float(self.config.neighbor_search_radius),
                    ],
                    device=self.device
                )
            
            # ===== CONTINUOUS FLOW THROUGH COMPONENTS (when system is running) =====
            if self.state.system_state == SystemState.RUNNING and self._discharge_open == 1:
                # Get component parameters
                feeder_pitch = self.assembly.feeder.params.screw_pitch
                airlock_length = self.assembly.airlock.params.rotor_length
                deagg_length = self.assembly.deagglomerator.params.housing_length
                
                # Get current rotation angles from components (for continuous animation sync)
                airlock_angle = self.assembly.airlock.get_rotor_angle()
                feeder_angle = self.assembly.feeder.get_screw_angle()
                deagg_angle = self.assembly.deagglomerator.get_rotor_angle()
                
                # Feeder inlet is below airlock outlet
                feeder_inlet = (
                    self.feeder_center[0],
                    self.airlock_center[1] - self.airlock_radius,  # Below airlock
                    self.feeder_center[2]
                )
                
                # Feeder axis direction (screw conveying direction)
                feeder_axis = (1.0, 0.0, 0.0)  # Along X axis
                
                wp.launch(
                    kernel=continuous_flow_kernel,
                    dim=n,
                    inputs=[
                        self.state.positions,
                        self.state.velocities,
                        self._particle_zones,
                        self.state.is_active,
                        n,
                        # Hopper
                        wp.vec3(float(self.hopper_center[0]), float(self.hopper_center[1]), float(self.hopper_center[2])),
                        float(self.hopper_bottom_radius),
                        float(self.hopper_center[1]),  # discharge_y = hopper bottom
                        # Airlock
                        wp.vec3(float(self.airlock_center[0]), float(self.airlock_center[1]), float(self.airlock_center[2])),
                        float(self.airlock_radius),
                        float(airlock_length),
                        float(self.state.airlock_rpm),
                        float(airlock_angle),
                        # Feeder
                        wp.vec3(float(feeder_inlet[0]), float(feeder_inlet[1]), float(feeder_inlet[2])),
                        wp.vec3(float(feeder_axis[0]), float(feeder_axis[1]), float(feeder_axis[2])),
                        float(self.feeder_radius),
                        float(self.feeder_length),
                        float(self.state.feeder_rpm),
                        float(feeder_pitch),
                        float(feeder_angle),
                        # Deagg
                        wp.vec3(float(self.deagg_center[0]), float(self.deagg_center[1]), float(self.deagg_center[2])),
                        float(self.deagg_radius),
                        float(deagg_length),
                        float(self.state.deagg_rpm),
                        float(deagg_angle),
                        # Exit
                        float(self.exit_y),
                        # Time
                        dt,
                        float(GRAVITY),
                        self._discharge_open,
                    ],
                    device=self.device
                )
                
                # Update component rotations for continuous animation
                self.assembly.airlock.update_rotation(dt, self.state.airlock_rpm)
                self.assembly.feeder.update_rotation(dt, self.state.feeder_rpm)
                self.assembly.deagglomerator.update_rotation(dt, self.state.deagg_rpm)
        
        # ===== UPDATE ZONE COUNTS (after flow simulation) =====
        if self.state.system_state == SystemState.RUNNING and hasattr(self, '_zone_counts'):
            self._update_zone_counts()
        
        # Update time
        self.state.time += dt
        self.state.step += 1
    
    def run(self, progress_callback=None):
        """Run the full simulation."""
        total_steps = self.config.num_steps
        
        self.start()
        
        for step in range(total_steps):
            self.step()
            
            if progress_callback and step % self.config.output_steps == 0:
                progress_callback(step, total_steps)
        
        wp.synchronize()
    
    def get_results(self) -> Dict[str, Any]:
        """Get simulation results."""
        # Get zone counts if available
        zone_counts = getattr(self, '_zone_counts', {
            'hopper': 0, 'airlock': 0, 'feeder': 0, 'deagg': 0, 'exit': 0
        })
        
        return {
            "time": self.state.time,
            "steps": self.state.step,
            "system_state": self.state.system_state.value,
            "airlock_rpm": self.state.airlock_rpm,
            "feeder_rpm": self.state.feeder_rpm,
            "deagg_rpm": self.state.deagg_rpm,
            "mass_flow_rate_kg_h": self.state.mass_flow_rate_kg_h,
            "hopper_mass_kg": self.state.hopper_mass_kg,
            "particles_injected": self.state.particles_injected,
            # Lid animation state
            "lid_state": self.state.lid_state.value,
            "lid_angle": self.state.lid_angle,
            "lid_angular_velocity": self.state.lid_angular_velocity,
            # Pouring state
            "pouring_state": self.state.pouring_state.value,
            "particles_poured": self.state.particles_poured,
            # Fill tracking
            "total_particles_to_pour": self.state.total_particles_to_pour,
            "particles_inside_hopper": self.state.particles_inside_hopper,
            "target_fill_mass_kg": self.state.target_fill_mass_kg,
            # Flow zone counts
            "zone_hopper": zone_counts.get('hopper', 0),
            "zone_airlock": zone_counts.get('airlock', 0),
            "zone_feeder": zone_counts.get('feeder', 0),
            "zone_deagg": zone_counts.get('deagg', 0),
            "zone_exit": zone_counts.get('exit', 0),
        }
    
    def get_particle_positions(self) -> np.ndarray:
        """Get current particle positions as numpy array."""
        if self.state.particles_injected == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return self.state.positions.numpy()[:self.state.particles_injected]
    
    def get_particle_velocities(self) -> np.ndarray:
        """Get current particle velocities as numpy array."""
        if self.state.particles_injected == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return self.state.velocities.numpy()[:self.state.particles_injected]
    
    def get_particle_diameters(self) -> np.ndarray:
        """Get particle diameters as numpy array."""
        if self.state.particles_injected == 0:
            return np.zeros(0, dtype=np.float32)
        return self.state.diameters.numpy()[:self.state.particles_injected]
    
    def get_active_particles(self) -> np.ndarray:
        """Get particle active state as numpy array."""
        if self.state.particles_injected == 0:
            return np.zeros(0, dtype=np.int32)
        return self.state.is_active.numpy()[:self.state.particles_injected]


# =============================================================================
# CLASSIFICATION SYSTEM SIMULATOR
# =============================================================================

@dataclass
class ClassificationConfig(BaseSimulationConfig):
    """Configuration for classification system simulation."""
    
    # Particle parameters
    num_particles: int = 10000       # Number of particles to simulate
    injection_duration: float = 0.1  # [s] Duration over which to inject
    
    # Flow parameters
    inlet_velocity: float = 15.0     # [m/s] Air inlet velocity
    
    # Separation targets
    target_cut_size_um: float = 20.0  # [μm] Target cut size


@dataclass
class ClassificationState(BaseSimulationState):
    """State for classification system simulation."""
    
    # Particle arrays (on device)
    positions: Optional[wp.array] = None
    velocities: Optional[wp.array] = None
    diameters: Optional[wp.array] = None
    is_active: Optional[wp.array] = None
    
    # Statistics
    particles_injected: int = 0
    particles_in_fines: int = 0       # Fine fraction (protein)
    particles_in_coarse: int = 0      # Coarse fraction (starch)
    particles_in_cyclone_1: int = 0   # Primary cyclone
    particles_in_cyclone_2: int = 0   # Secondary cyclone
    particles_in_cyclone_3: int = 0   # Tertiary cyclone
    particles_in_bag_filter: int = 0  # Bag filter
    particles_active: int = 0
    
    # Size tracking
    fines_diameters: List[float] = field(default_factory=list)
    coarse_diameters: List[float] = field(default_factory=list)


class ClassificationSystemSimulator:
    """
    Simulator for the classification system.
    
    Simulates particle separation through:
    - Venturi Eductor (entrainment)
    - Zigzag Classifier (primary separation)
    - Multi-Cyclone System (staged collection)
    - Bag Filter (fine capture)
    
    Tracks particles and computes separation efficiency.
    """
    
    def __init__(
        self,
        assembly: Any,  # ClassificationSystemAssembly
        config: ClassificationConfig = None,
        material: Any = None,  # Optional ParticleMaterial
    ):
        """
        Initialize classification system simulator.
        
        Args:
            assembly: ClassificationSystemAssembly instance
            config: Simulation configuration
            material: Particle material definition (optional)
        """
        self.assembly = assembly
        self.config = config or ClassificationConfig()
        self.material = material
        self.state = ClassificationState()
        
        # Initialize Warp
        wp.init()
        self.device = self.config.device
        
        # Setup system
        self._setup_system_parameters()
        self._allocate_arrays()
        self._setup_flow_field()
    
    def _setup_system_parameters(self):
        """Extract parameters from assembly."""
        self.venturi = self.assembly.venturi
        self.zigzag = self.assembly.zigzag
        self.multi_cyclone = self.assembly.multi_cyclone
        self.bag_filter = self.assembly.bag_filter
        
        # Get component positions
        self.component_positions = self.assembly.get_component_positions()
        
        # Get system bounds for domain setup
        self.bounds_min, self.bounds_max = self.assembly.get_bounds()
    
    def _allocate_arrays(self):
        """Pre-allocate particle arrays."""
        n = self.config.num_particles
        
        self.state.positions = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.state.velocities = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.state.diameters = wp.zeros(n, dtype=float, device=self.device)
        self.state.is_active = wp.zeros(n, dtype=wp.int32, device=self.device)
        
        # Temporary arrays
        self._accelerations = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self._fluid_velocities = wp.zeros(n, dtype=wp.vec3, device=self.device)
    
    def _setup_flow_field(self):
        """Setup analytical flow field for the classification system."""
        # Store flow parameters for the analytical model
        self.flow_params = {
            'inlet_velocity': self.config.inlet_velocity,
            'fluid_density': AirProperties.DENSITY,
            'fluid_viscosity': AirProperties.DYNAMIC_VISCOSITY,
        }
        
        # Venturi parameters
        venturi_pos = np.array(self.component_positions['venturi'])
        self.venturi_center = wp.vec3(venturi_pos[0], venturi_pos[1], venturi_pos[2])
        self.venturi_inlet_d = self.venturi.params.inlet_diameter
        self.venturi_throat_d = self.venturi.params.throat_diameter
        
        # Zigzag parameters
        zigzag_pos = np.array(self.component_positions['zigzag'])
        self.zigzag_center = wp.vec3(zigzag_pos[0], zigzag_pos[1], zigzag_pos[2])
        self.zigzag_width = self.zigzag.params.channel_width
        self.zigzag_height = self.zigzag.params.total_height
        
        # Cyclone parameters (primary)
        cyclone_pos = np.array(self.component_positions['multi_cyclone'])
        self.cyclone_center = wp.vec3(cyclone_pos[0], cyclone_pos[1], cyclone_pos[2])
    
    def start(self):
        """Start the classification system."""
        self.state.system_state = SystemState.RUNNING
    
    def stop(self):
        """Stop the classification system."""
        self.state.system_state = SystemState.OFF
    
    def inject_particles(self, n_inject: int):
        """Inject particles at the venturi inlet."""
        if self.state.particles_injected >= self.config.num_particles:
            return
        
        n_inject = min(n_inject, self.config.num_particles - self.state.particles_injected)
        
        # Generate particle sizes
        if self.material is not None:
            diameters = self.material.sample_diameters(n_inject, seed=self.state.step + 42)
        else:
            # Default: log-normal distribution
            rng = np.random.default_rng(self.state.step + 42)
            mean_d = 30e-6  # 30 microns
            diameters = rng.lognormal(np.log(mean_d), 0.7, n_inject).astype(np.float32)
            diameters = np.clip(diameters, 1e-6, 200e-6)
        
        # Get venturi solids inlet position
        venturi_pos = np.array(self.component_positions['venturi'])
        solids_inlet = self.venturi.ports['solids_inlet']
        inlet_pos = venturi_pos + np.array(solids_inlet.position)
        inlet_dir = np.array(solids_inlet.direction)
        inlet_radius = solids_inlet.diameter / 2 * 0.8
        
        # Generate positions at inlet
        rng = np.random.default_rng(self.state.step + 123)
        positions = np.zeros((n_inject, 3), dtype=np.float32)
        velocities = np.zeros((n_inject, 3), dtype=np.float32)
        
        for i in range(n_inject):
            # Random position within inlet area
            r = rng.random() * inlet_radius
            theta = rng.random() * 2 * np.pi
            
            # Calculate perpendicular directions
            if abs(inlet_dir[1]) < 0.9:
                perp1 = np.cross(inlet_dir, [0, 1, 0])
            else:
                perp1 = np.cross(inlet_dir, [1, 0, 0])
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(inlet_dir, perp1)
            
            positions[i] = inlet_pos + r * (np.cos(theta) * perp1 + np.sin(theta) * perp2)
            
            # Initial velocity in inlet direction
            velocities[i] = inlet_dir * 2.0  # 2 m/s entry velocity
        
        # Copy to device
        start_idx = self.state.particles_injected
        
        positions_wp = wp.array(positions, dtype=wp.vec3, device=self.device)
        velocities_wp = wp.array(velocities, dtype=wp.vec3, device=self.device)
        diameters_wp = wp.array(diameters.astype(np.float32), dtype=float, device=self.device)
        
        wp.copy(self.state.positions, positions_wp, dest_offset=start_idx, count=n_inject)
        wp.copy(self.state.velocities, velocities_wp, dest_offset=start_idx, count=n_inject)
        wp.copy(self.state.diameters, diameters_wp, dest_offset=start_idx, count=n_inject)
        
        active_flags = wp.array(np.ones(n_inject, dtype=np.int32), dtype=wp.int32, device=self.device)
        wp.copy(self.state.is_active, active_flags, dest_offset=start_idx, count=n_inject)
        
        self.state.particles_injected += n_inject
        self.state.particles_active += n_inject
    
    def step(self):
        """Advance simulation by one time step."""
        dt = self.config.dt
        
        # Inject particles during injection phase
        if self.state.time < self.config.injection_duration:
            inject_rate = self.config.num_particles / self.config.injection_duration
            n_inject = int(inject_rate * dt) + (1 if np.random.random() < (inject_rate * dt % 1) else 0)
            if n_inject > 0:
                self.inject_particles(n_inject)
        
        # Skip if no active particles
        if self.state.particles_active == 0 and self.state.particles_injected >= self.config.num_particles:
            self.state.time += dt
            self.state.step += 1
            return
        
        n = self.state.particles_injected
        if n == 0:
            self.state.time += dt
            self.state.step += 1
            return
        
        # Compute fluid velocities at particle positions
        wp.launch(
            kernel=classification_fluid_velocity_kernel,
            dim=n,
            inputs=[
                self.state.positions,
                self._fluid_velocities,
                self.state.is_active,
                self.venturi_center,
                self.zigzag_center,
                self.cyclone_center,
                self.venturi_inlet_d,
                float(self.config.inlet_velocity),
            ],
            device=self.device
        )
        
        # Compute accelerations
        wp.launch(
            kernel=classification_particle_accelerations,
            dim=n,
            inputs=[
                self.state.positions,
                self.state.velocities,
                self.state.diameters,
                self._fluid_velocities,
                self.state.is_active,
                self._accelerations,
                2500.0,  # Particle density (flour ~2500 kg/m³)
                AirProperties.DENSITY,
                AirProperties.DYNAMIC_VISCOSITY,
                float(GRAVITY),
                self.config.include_drag,
                self.config.include_gravity,
            ],
            device=self.device
        )
        
        # Integrate
        wp.launch(
            kernel=integrate_euler_kernel,
            dim=n,
            inputs=[
                self.state.positions,
                self.state.velocities,
                self._accelerations,
                self.state.is_active,
                dt,
            ],
            device=self.device
        )
        
        # Check particle collection zones
        wp.launch(
            kernel=classification_check_collection,
            dim=n,
            inputs=[
                self.state.positions,
                self.state.is_active,
                self.zigzag_center,
                self.zigzag_height,
                self.cyclone_center,
            ],
            device=self.device
        )
        
        # Update time
        self.state.time += dt
        self.state.step += 1
    
    def run(self, progress_callback=None):
        """Run the full simulation."""
        total_steps = self.config.num_steps
        self.state.system_state = SystemState.RUNNING
        
        for step in range(total_steps):
            self.step()
            
            if progress_callback and step % self.config.output_steps == 0:
                progress_callback(step, total_steps)
        
        wp.synchronize()
    
    def get_results(self) -> Dict[str, Any]:
        """Get simulation results."""
        # Copy data back to CPU
        is_active = self.state.is_active.numpy()
        diameters = self.state.diameters.numpy()
        
        # Count particles in each zone
        # is_active encoding: 1=active, -1=coarse, -2=fines, -3=cyclone1, -4=cyclone2, -5=cyclone3, -6=bagfilter
        n_coarse = np.sum(is_active == -1)
        n_fines = np.sum(is_active == -2)
        n_cyclone1 = np.sum(is_active == -3)
        n_cyclone2 = np.sum(is_active == -4)
        n_cyclone3 = np.sum(is_active == -5)
        n_bagfilter = np.sum(is_active == -6)
        n_active = np.sum(is_active == 1)
        
        # Get diameters by fraction
        coarse_d = diameters[is_active == -1] if n_coarse > 0 else np.array([])
        fines_d = diameters[is_active == -2] if n_fines > 0 else np.array([])
        
        return {
            "time": self.state.time,
            "steps": self.state.step,
            "particles_injected": self.state.particles_injected,
            "particles_active": int(n_active),
            "particles_coarse": int(n_coarse),
            "particles_fines": int(n_fines),
            "particles_cyclone_1": int(n_cyclone1),
            "particles_cyclone_2": int(n_cyclone2),
            "particles_cyclone_3": int(n_cyclone3),
            "particles_bag_filter": int(n_bagfilter),
            "separation_efficiency": float(n_fines / max(1, n_coarse + n_fines)),
            "mean_coarse_diameter_um": float(np.mean(coarse_d) * 1e6) if len(coarse_d) > 0 else 0.0,
            "mean_fines_diameter_um": float(np.mean(fines_d) * 1e6) if len(fines_d) > 0 else 0.0,
        }


# =============================================================================
# COMPLETE SYSTEM SIMULATOR
# =============================================================================

@dataclass
class CompleteSystemConfig(BaseSimulationConfig):
    """Configuration for complete system simulation."""
    
    # Air system config
    blower_rpm: float = 3000.0
    damper_positions: List[float] = field(default_factory=lambda: [1.0, 1.0])
    
    # Feed system config
    feed_rate_kg_h: float = 500.0
    airlock_rpm: float = 20.0
    feeder_rpm: float = 60.0
    deagg_rpm: float = 1500.0
    
    # Classification config
    num_particles: int = 10000
    injection_duration: float = 0.1
    inlet_velocity: float = 15.0
    
    # Startup sequence
    startup_sequence: List[str] = field(default_factory=lambda: ["air", "classification", "feed"])
    startup_delay: float = 0.5  # [s] Delay between system startups


@dataclass
class CompleteSystemState(BaseSimulationState):
    """State for complete system simulation."""
    
    # Subsystem states
    air_system_state: SystemState = SystemState.OFF
    feed_system_state: SystemState = SystemState.OFF
    classification_state: SystemState = SystemState.OFF
    
    # Flow coupling
    total_flow_rate_m3_h: float = 0.0
    system_pressure_Pa: float = 0.0
    
    # Material tracking
    feed_rate_kg_h: float = 0.0
    particles_in_system: int = 0
    
    # Energy
    total_power_kW: float = 0.0


class CompleteSystemSimulator:
    """
    Complete air classifier system simulator.
    
    Integrates and couples all three subsystems:
    - Air System: Provides motive force
    - Feed System: Delivers material
    - Classification System: Separates particles
    
    Handles startup sequence, inter-system coupling, and complete
    material balance tracking.
    """
    
    def __init__(
        self,
        assembly: Any,  # CompleteClassifierAssembly
        config: CompleteSystemConfig = None,
        material: Any = None,  # Optional ParticleMaterial
    ):
        """
        Initialize complete system simulator.
        
        Args:
            assembly: CompleteClassifierAssembly instance
            config: Simulation configuration
            material: Particle material definition
        """
        self.assembly = assembly
        self.config = config or CompleteSystemConfig()
        self.material = material
        self.state = CompleteSystemState()
        
        # Initialize Warp
        wp.init()
        self.device = self.config.device
        
        # Create subsystem simulators
        self._create_subsystem_simulators()
        
        # Startup tracking
        self._startup_index = 0
        self._startup_timer = 0.0
    
    def _create_subsystem_simulators(self):
        """Create simulators for each subsystem."""
        # Get subsystems from assembly
        air_system = self.assembly.get_subsystem('air_system')
        feed_system = self.assembly.get_subsystem('feed_system')
        classification = self.assembly.get_subsystem('classification')
        
        # Air system simulator
        if air_system is not None:
            air_config = AirSystemConfig(
                dt=self.config.dt,
                duration=self.config.duration,
                blower_rpm=self.config.blower_rpm,
                damper_positions=self.config.damper_positions,
                device=self.config.device,
            )
            self.air_simulator = AirSystemSimulator(air_system, air_config)
        else:
            self.air_simulator = None
        
        # Feed system simulator
        if feed_system is not None:
            feed_config = FeedSystemConfig(
                dt=self.config.dt,
                duration=self.config.duration,
                feed_rate_kg_h=self.config.feed_rate_kg_h,
                airlock_rpm=self.config.airlock_rpm,
                feeder_rpm=self.config.feeder_rpm,
                deagg_rpm=self.config.deagg_rpm,
                device=self.config.device,
            )
            self.feed_simulator = FeedSystemSimulator(feed_system, feed_config)
        else:
            self.feed_simulator = None
        
        # Classification simulator
        if classification is not None:
            class_config = ClassificationConfig(
                dt=self.config.dt,
                duration=self.config.duration,
                num_particles=self.config.num_particles,
                injection_duration=self.config.injection_duration,
                inlet_velocity=self.config.inlet_velocity,
                device=self.config.device,
            )
            self.classification_simulator = ClassificationSystemSimulator(
                classification, class_config, self.material
            )
        else:
            self.classification_simulator = None
    
    def start(self):
        """Start the complete system (initiates startup sequence)."""
        self.state.system_state = SystemState.STARTING
        self._startup_index = 0
        self._startup_timer = 0.0
        # Start the first system immediately
        if self.config.startup_sequence:
            first_system = self.config.startup_sequence[0]
            self.start_system(first_system)
            self._startup_index = 1
    
    def start_system(self, system_name: str):
        """Start a specific subsystem."""
        if system_name == "air" and self.air_simulator:
            self.air_simulator.start()
            self.state.air_system_state = SystemState.STARTING
        elif system_name == "feed" and self.feed_simulator:
            self.feed_simulator.start()
            self.state.feed_system_state = SystemState.STARTING
        elif system_name == "classification" and self.classification_simulator:
            self.classification_simulator.start()
            self.state.classification_state = SystemState.RUNNING
    
    def stop_all(self):
        """Stop all subsystems."""
        if self.air_simulator:
            self.air_simulator.stop()
        if self.feed_simulator:
            self.feed_simulator.stop()
        if self.classification_simulator:
            self.classification_simulator.stop()
    
    def step(self):
        """Advance simulation by one time step."""
        dt = self.config.dt
        
        # Handle startup sequence
        if self._startup_index < len(self.config.startup_sequence):
            self._startup_timer += dt
            
            if self._startup_timer >= self.config.startup_delay:
                system_name = self.config.startup_sequence[self._startup_index]
                self.start_system(system_name)
                self._startup_index += 1
                self._startup_timer = 0.0
        
        # Step each subsystem
        if self.air_simulator:
            self.air_simulator.step()
            self.state.air_system_state = self.air_simulator.state.system_state
            
        if self.feed_simulator:
            self.feed_simulator.step()
            self.state.feed_system_state = self.feed_simulator.state.system_state
            
        if self.classification_simulator:
            # Couple air velocity to classification
            if self.air_simulator and self.air_simulator.state.system_state == SystemState.RUNNING:
                # Scale inlet velocity based on actual flow rate
                flow_ratio = self.air_simulator.state.flow_rate_m3_h / self.air_simulator.config.design_flow_rate_m3_h
                self.classification_simulator.config.inlet_velocity = self.config.inlet_velocity * flow_ratio
            
            self.classification_simulator.step()
            self.state.classification_state = self.classification_simulator.state.system_state
        
        # Update coupled state
        if self.air_simulator:
            self.state.total_flow_rate_m3_h = self.air_simulator.state.flow_rate_m3_h
            self.state.system_pressure_Pa = self.air_simulator.state.pressure_Pa
            self.state.total_power_kW = self.air_simulator.state.power_consumption_kW
            
        if self.feed_simulator:
            self.state.feed_rate_kg_h = self.feed_simulator.state.mass_flow_rate_kg_h
            
        if self.classification_simulator:
            self.state.particles_in_system = self.classification_simulator.state.particles_active
        
        # Update time
        self.state.time += dt
        self.state.step += 1
        
        # Update system state
        if (self.state.air_system_state == SystemState.RUNNING and
            self.state.classification_state == SystemState.RUNNING):
            self.state.system_state = SystemState.RUNNING
    
    def run(self, progress_callback=None):
        """Run the full simulation."""
        total_steps = self.config.num_steps
        
        # Auto-start the system
        self.start()
        
        for step in range(total_steps):
            self.step()
            
            if progress_callback and step % self.config.output_steps == 0:
                progress_callback(step, total_steps)
        
        wp.synchronize()
    
    def get_results(self) -> Dict[str, Any]:
        """Get comprehensive simulation results."""
        results = {
            "time": self.state.time,
            "steps": self.state.step,
            "system_state": self.state.system_state.value,
            "total_flow_rate_m3_h": self.state.total_flow_rate_m3_h,
            "system_pressure_Pa": self.state.system_pressure_Pa,
            "total_power_kW": self.state.total_power_kW,
            "feed_rate_kg_h": self.state.feed_rate_kg_h,
        }
        
        # Add subsystem results
        if self.air_simulator:
            results["air_system"] = self.air_simulator.get_results()
            
        if self.feed_simulator:
            results["feed_system"] = self.feed_simulator.get_results()
            
        if self.classification_simulator:
            results["classification"] = self.classification_simulator.get_results()
        
        return results


# =============================================================================
# WARP KERNELS
# =============================================================================

@wp.kernel
def feed_particle_update_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    diameters: wp.array(dtype=float),
    is_active: wp.array(dtype=wp.int32),
    dt: float,
    g: float,
    rho_f: float,
    mu_f: float,
):
    """Update particle positions in feed system with gravity and drag."""
    tid = wp.tid()
    
    if is_active[tid] != 1:
        return
    
    pos = positions[tid]
    vel = velocities[tid]
    d = diameters[tid]
    
    # Gravity
    acc = wp.vec3(0.0, -g, 0.0)
    
    # Simple Stokes drag (assume low Re in feed system)
    if d > 1.0e-8:
        drag_coeff = 18.0 * mu_f / (2500.0 * d * d)  # Stokes drag
        acc = acc - vel * drag_coeff
    
    # Integrate
    new_vel = vel + acc * dt
    new_pos = pos + new_vel * dt
    
    velocities[tid] = new_vel
    positions[tid] = new_pos


@wp.kernel
def classification_fluid_velocity_kernel(
    positions: wp.array(dtype=wp.vec3),
    fluid_velocities: wp.array(dtype=wp.vec3),
    is_active: wp.array(dtype=wp.int32),
    venturi_center: wp.vec3,
    zigzag_center: wp.vec3,
    cyclone_center: wp.vec3,
    venturi_d: float,
    inlet_velocity: float,
):
    """Compute fluid velocities at particle positions in classification system."""
    tid = wp.tid()
    
    if is_active[tid] != 1:
        fluid_velocities[tid] = wp.vec3(0.0, 0.0, 0.0)
        return
    
    pos = positions[tid]
    
    # Determine which region particle is in and apply appropriate flow model
    # Region detection based on Y position relative to component centers
    
    y_venturi = venturi_center[1]
    y_zigzag = zigzag_center[1]
    y_cyclone = cyclone_center[1]
    
    # Venturi region: strong vertical upward flow
    if pos[1] < y_zigzag - 0.5:
        # Accelerated flow through venturi throat
        v_mag = inlet_velocity * 1.5  # Throat acceleration
        fluid_velocities[tid] = wp.vec3(0.0, v_mag, 0.0)
        return
    
    # Zigzag region: oscillating upward flow
    if pos[1] < y_zigzag + 1.0:
        # Zigzag creates sinusoidal horizontal component
        phase = pos[1] * 10.0  # ~10 oscillations per meter
        v_horiz = inlet_velocity * 0.3 * wp.sin(phase)
        v_vert = inlet_velocity * 0.8
        fluid_velocities[tid] = wp.vec3(v_horiz, v_vert, 0.0)
        return
    
    # Cyclone region: swirling flow (Rankine vortex)
    dx = pos[0] - cyclone_center[0]
    dz = pos[2] - cyclone_center[2]
    r = wp.sqrt(dx * dx + dz * dz)
    
    cyclone_r = 0.15  # Approximate cyclone radius
    
    if r > 0.001:
        # Tangential velocity (free vortex outside core)
        v_tan = inlet_velocity * 0.8 * cyclone_r / wp.max(r, cyclone_r * 0.3)
        
        # Radial velocity (inward)
        v_rad = -inlet_velocity * 0.1
        
        # Axial velocity (downward near wall, upward in center)
        if r < cyclone_r * 0.5:
            v_axial = inlet_velocity * 0.5  # Upward in core
        else:
            v_axial = -inlet_velocity * 0.3  # Downward near wall
        
        # Convert to Cartesian
        cos_theta = dx / r
        sin_theta = dz / r
        
        vx = -v_tan * sin_theta + v_rad * cos_theta
        vz = v_tan * cos_theta + v_rad * sin_theta
        
        fluid_velocities[tid] = wp.vec3(vx, v_axial, vz)
    else:
        fluid_velocities[tid] = wp.vec3(0.0, inlet_velocity * 0.5, 0.0)


@wp.kernel
def classification_particle_accelerations(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    diameters: wp.array(dtype=float),
    fluid_velocities: wp.array(dtype=wp.vec3),
    is_active: wp.array(dtype=wp.int32),
    accelerations: wp.array(dtype=wp.vec3),
    rho_p: float,
    rho_f: float,
    mu_f: float,
    g: float,
    include_drag: bool,
    include_gravity: bool,
):
    """Compute particle accelerations in classification system."""
    tid = wp.tid()
    
    if is_active[tid] != 1:
        accelerations[tid] = wp.vec3(0.0, 0.0, 0.0)
        return
    
    pos = positions[tid]
    vel = velocities[tid]
    d = diameters[tid]
    v_f = fluid_velocities[tid]
    
    acc = wp.vec3(0.0, 0.0, 0.0)
    
    # Drag force (Schiller-Naumann)
    if include_drag:
        v_rel = v_f - vel
        v_rel_mag = wp.length(v_rel)
        eps = 1.0e-10
        
        if v_rel_mag > eps and d > eps:
            # Reynolds number
            Re = rho_f * v_rel_mag * d / mu_f
            
            # Drag coefficient
            if Re < eps:
                Cd = 24.0 / eps
            else:
                Cd = (24.0 / Re) * (1.0 + 0.15 * wp.pow(Re, 0.687))
            
            # Particle properties
            volume = (3.141592653589793 / 6.0) * d * d * d
            mass = rho_p * volume
            A_p = (3.141592653589793 / 4.0) * d * d
            
            # Drag acceleration
            a_drag_mag = 0.5 * Cd * rho_f * A_p * v_rel_mag * v_rel_mag / mass
            acc = acc + v_rel * (a_drag_mag / v_rel_mag)
    
    # Gravity with buoyancy
    if include_gravity:
        a_grav = (1.0 - rho_f / rho_p) * g
        acc = acc + wp.vec3(0.0, -a_grav, 0.0)
    
    accelerations[tid] = acc


@wp.kernel
def integrate_euler_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    accelerations: wp.array(dtype=wp.vec3),
    is_active: wp.array(dtype=wp.int32),
    dt: float,
):
    """Euler integration step."""
    tid = wp.tid()
    
    if is_active[tid] != 1:
        return
    
    vel = velocities[tid]
    acc = accelerations[tid]
    pos = positions[tid]
    
    new_vel = vel + acc * dt
    new_pos = pos + new_vel * dt
    
    velocities[tid] = new_vel
    positions[tid] = new_pos


@wp.kernel
def classification_check_collection(
    positions: wp.array(dtype=wp.vec3),
    is_active: wp.array(dtype=wp.int32),
    zigzag_center: wp.vec3,
    zigzag_height: float,
    cyclone_center: wp.vec3,
):
    """Check if particles have been collected in various zones."""
    tid = wp.tid()
    
    if is_active[tid] != 1:
        return
    
    pos = positions[tid]
    
    # Coarse outlet (below zigzag)
    if pos[1] < zigzag_center[1] - zigzag_height:
        is_active[tid] = -1  # Coarse fraction
        return
    
    # Check cyclone collection (particles that fall through dust outlet)
    # Primary cyclone
    dx = pos[0] - cyclone_center[0]
    dz = pos[2] - cyclone_center[2]
    r = wp.sqrt(dx * dx + dz * dz)
    
    if pos[1] < cyclone_center[1] - 0.5 and r < 0.1:
        is_active[tid] = -3  # Primary cyclone
        return
    
    # Fines (top exit)
    if pos[1] > zigzag_center[1] + zigzag_height + 1.0:
        is_active[tid] = -2  # Fines fraction
        return


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_air_system_simulator(
    assembly: Any,
    blower_rpm: float = 3000.0,
    damper_positions: List[float] = None,
    device: str = "cuda",
) -> AirSystemSimulator:
    """
    Create an air system simulator.
    
    Args:
        assembly: AirSystemAssembly instance
        blower_rpm: Blower operating speed [RPM]
        damper_positions: List of damper positions (0-1)
        device: Warp device
        
    Returns:
        AirSystemSimulator instance
    """
    config = AirSystemConfig(
        blower_rpm=blower_rpm,
        damper_positions=damper_positions or [1.0, 1.0],
        device=device,
    )
    return AirSystemSimulator(assembly, config)


def create_feed_system_simulator(
    assembly: Any,
    feed_rate_kg_h: float = 500.0,
    device: str = "cuda",
) -> FeedSystemSimulator:
    """
    Create a feed system simulator.
    
    Args:
        assembly: FeedSystemAssembly instance
        feed_rate_kg_h: Target feed rate [kg/h]
        device: Warp device
        
    Returns:
        FeedSystemSimulator instance
    """
    config = FeedSystemConfig(
        feed_rate_kg_h=feed_rate_kg_h,
        device=device,
    )
    return FeedSystemSimulator(assembly, config)


def create_classification_simulator(
    assembly: Any,
    num_particles: int = 10000,
    inlet_velocity: float = 15.0,
    material: Any = None,
    device: str = "cuda",
) -> ClassificationSystemSimulator:
    """
    Create a classification system simulator.
    
    Args:
        assembly: ClassificationSystemAssembly instance
        num_particles: Number of particles to simulate
        inlet_velocity: Air inlet velocity [m/s]
        material: ParticleMaterial instance
        device: Warp device
        
    Returns:
        ClassificationSystemSimulator instance
    """
    config = ClassificationConfig(
        num_particles=num_particles,
        inlet_velocity=inlet_velocity,
        device=device,
    )
    return ClassificationSystemSimulator(assembly, config, material)


def create_complete_system_simulator(
    assembly: Any,
    blower_rpm: float = 3000.0,
    feed_rate_kg_h: float = 500.0,
    num_particles: int = 10000,
    material: Any = None,
    device: str = "cuda",
) -> CompleteSystemSimulator:
    """
    Create a complete system simulator.
    
    Args:
        assembly: CompleteClassifierAssembly instance
        blower_rpm: Blower operating speed [RPM]
        feed_rate_kg_h: Target feed rate [kg/h]
        num_particles: Number of particles to simulate
        material: ParticleMaterial instance
        device: Warp device
        
    Returns:
        CompleteSystemSimulator instance
    """
    config = CompleteSystemConfig(
        blower_rpm=blower_rpm,
        feed_rate_kg_h=feed_rate_kg_h,
        num_particles=num_particles,
        device=device,
    )
    return CompleteSystemSimulator(assembly, config, material)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Entry point for command-line usage."""
    import sys
    
    print("=" * 70)
    print("Air Classifier System Simulators")
    print("=" * 70)
    print("\nAvailable simulators:")
    print("  1. AirSystemSimulator       - Blower, filter, dampers")
    print("  2. FeedSystemSimulator      - Hopper, airlock, feeder, deagglomerator")
    print("  3. ClassificationSimulator  - Venturi, zigzag, cyclones, bag filter")
    print("  4. CompleteSystemSimulator  - Full integrated system")
    print("\nUsage:")
    print("  from airclassifier.simulation.simulator import (")
    print("      create_air_system_simulator,")
    print("      create_feed_system_simulator,")
    print("      create_classification_simulator,")
    print("      create_complete_system_simulator,")
    print("  )")
    print("\nExample:")
    print("  from airclassifier.geometry.assembly import create_standard_air_system")
    print("  assembly = create_standard_air_system()")
    print("  sim = create_air_system_simulator(assembly)")
    print("  sim.run()")
    print("  results = sim.get_results()")
    print("=" * 70)


if __name__ == "__main__":
    main()
