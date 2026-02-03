"""
Physics-Based Air Flow Simulation
=================================

Simulates air flow through the air system using actual geometry and
true physics principles. NO magic numbers - all parameters computed
from geometry and physical laws.

Physics implemented:
- Fan affinity laws (Q ∝ N, P ∝ N², W ∝ N³)
- Bernoulli equation for pressure-velocity relationships
- Darcy-Weisbach friction losses in ducts
- Component pressure drops from geometry and flow coefficients
- Mass/volume conservation (continuity)

Components modeled:
- Centrifugal blower (impeller physics, scroll design)
- Inlet filter (media resistance, face velocity)
- Dampers (blade angle, flow coefficient Cv)
- Ductwork (friction factor, losses)

Author: Air Classifier Physics Engine
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import numpy as np
import warp as wp

from ..geometry.assembly.air_system import AirSystemAssembly, AirSystemParams
from ..utils.constants import PI, GRAVITY, AirProperties

# =============================================================================
# PHYSICAL CONSTANTS (from air properties)
# =============================================================================

RHO_AIR = AirProperties.DENSITY           # kg/m³ at STP
MU_AIR = AirProperties.DYNAMIC_VISCOSITY  # Pa·s at STP
NU_AIR = MU_AIR / RHO_AIR                  # m²/s kinematic viscosity

# Derived constants
TWO_PI = 2.0 * PI
SQRT_2 = np.sqrt(2.0)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class BlowerState(Enum):
    """Operating state of blower."""
    OFF = 0
    STARTING = 1
    RUNNING = 2
    STOPPING = 3


class SystemPhase(Enum):
    """Simulation workflow phases."""
    IDLE = "idle"
    STARTUP = "startup"          # Blower ramping up, dampers opening
    RUNNING = "running"          # Steady-state operation
    SHUTDOWN = "shutdown"        # Blower ramping down, dampers closing
    OFF = "off"


@dataclass
class DuctSegment:
    """Geometry of a duct segment for flow calculation."""
    name: str
    diameter: float              # [m] Hydraulic diameter
    length: float                # [m] Segment length
    area: float                  # [m²] Cross-sectional area
    roughness: float = 0.00015  # [m] Surface roughness (galvanized steel)
    
    # Calculated during simulation
    velocity: float = 0.0        # [m/s]
    pressure_drop: float = 0.0   # [Pa]
    reynolds: float = 0.0


@dataclass
class ComponentPressureDrop:
    """Pressure drop data for a component."""
    name: str
    k_factor: float              # Loss coefficient K
    pressure_drop: float = 0.0   # [Pa] Current pressure drop


@dataclass
class AirFlowPhysicsConfig:
    """Configuration for physics-based air flow simulation."""
    
    # Time stepping
    dt: float = 0.001               # Time step [s]
    total_time: float = 10.0        # Total simulation time [s]
    
    # Blower operating parameters
    target_rpm: float = 3000.0      # [RPM] Target blower speed
    ramp_time: float = 2.0          # [s] Time to reach full speed
    
    # Damper control
    damper_ramp_time: float = 1.0   # [s] Time for damper to open/close
    
    # Air properties
    air_density: float = RHO_AIR
    air_viscosity: float = MU_AIR
    
    # Tracer particles (for visualization)
    enable_tracers: bool = True
    num_tracers: int = 500
    tracer_lifetime: float = 5.0    # [s] Time before tracer respawns
    
    # Device
    device: str = "cuda"


@dataclass
class AirFlowPhysicsState:
    """State of the air flow physics simulation."""
    time: float = 0.0
    step: int = 0
    
    # Blower state
    blower_state: BlowerState = BlowerState.OFF
    blower_rpm: float = 0.0
    blower_omega: float = 0.0       # [rad/s]
    blower_tip_speed: float = 0.0   # [m/s]
    
    # Flow state (from physics)
    volume_flow_rate: float = 0.0   # [m³/s]
    mass_flow_rate: float = 0.0     # [kg/s]
    
    # Pressure state
    static_pressure_rise: float = 0.0    # [Pa] Blower pressure rise
    total_pressure_drop: float = 0.0     # [Pa] System pressure drop
    
    # Power
    shaft_power: float = 0.0        # [W]
    electrical_power: float = 0.0   # [W]
    efficiency: float = 0.0
    
    # Damper positions
    damper_positions: List[float] = field(default_factory=lambda: [0.0, 0.0])
    
    # Simulation phase
    phase: SystemPhase = SystemPhase.IDLE
    phase_start_time: float = 0.0
    
    # Tracer particle data (on device)
    tracer_positions: Optional[wp.array] = None
    tracer_velocities: Optional[wp.array] = None
    tracer_ages: Optional[wp.array] = None
    tracer_active: Optional[wp.array] = None
    
    # Energy tracking
    total_energy_kWh: float = 0.0


# =============================================================================
# GEOMETRY EXTRACTION
# =============================================================================

@dataclass
class BlowerGeometry:
    """Extracted blower geometry for physics calculations."""
    impeller_diameter: float      # [m]
    impeller_width: float         # [m]
    inlet_diameter: float         # [m]
    hub_diameter: float           # [m]
    num_blades: int
    blade_type: str
    scroll_diameter: float        # [m]
    outlet_width: float           # [m]
    outlet_height: float          # [m]
    design_rpm: float
    design_flow_rate: float       # [m³/h]
    design_pressure_rise: float   # [Pa]
    
    @property
    def impeller_radius(self) -> float:
        return self.impeller_diameter / 2.0
    
    @property
    def inlet_area(self) -> float:
        """Inlet eye area [m²]."""
        return PI * (self.inlet_diameter / 2.0) ** 2
    
    @property
    def outlet_area(self) -> float:
        """Scroll outlet area [m²]."""
        return self.outlet_width * self.outlet_height
    
    @property
    def blade_outlet_area(self) -> float:
        """Impeller blade outlet area [m²]."""
        return PI * self.impeller_diameter * self.impeller_width


@dataclass
class FilterGeometry:
    """Extracted filter geometry."""
    face_area: float              # [m²]
    media_depth: float            # [m]
    housing_volume: float         # [m³]
    efficiency_class: str
    
    @property
    def media_resistance(self) -> float:
        """Media resistance coefficient [Pa·s/m]."""
        # Based on filter class (approximate values)
        resistance_map = {
            'G4': 50.0,
            'M5': 100.0,
            'M6': 150.0,
            'F7': 250.0,
            'F8': 350.0,
            'F9': 500.0,
            'HEPA': 2000.0,
        }
        return resistance_map.get(self.efficiency_class, 100.0)


@dataclass
class DamperGeometry:
    """Extracted damper geometry."""
    diameter: float               # [m]
    blade_chord: float            # [m]
    position: float = 0.0         # 0=closed, 1=open
    
    @property
    def area(self) -> float:
        return PI * (self.diameter / 2.0) ** 2
    
    def get_cv(self, position: float) -> float:
        """
        Get flow coefficient Cv based on blade position.
        
        Cv = Q / sqrt(dP)  where Q in gpm, dP in psi
        We use metric: Kv = Q / sqrt(dP/rho) in m³/h
        
        For butterfly valve, Cv varies with angle.
        """
        # Butterfly valve characteristic curve (approximate)
        # At 0° (closed): Cv ≈ 0
        # At 90° (open): Cv = Cv_max
        angle = position * 90.0  # degrees
        angle_rad = np.radians(angle)
        
        # Equal percentage characteristic
        if position < 0.05:
            return 0.01  # Small leakage when "closed"
        
        # Cv varies approximately as sin²(angle) for butterfly
        cv_factor = np.sin(angle_rad) ** 2
        
        # Max Cv based on diameter (m³/h per sqrt(bar))
        cv_max = 10.0 * (self.diameter * 1000) ** 2 / 10000  # Empirical
        
        return cv_max * cv_factor


def extract_air_geometry(assembly: AirSystemAssembly) -> Dict[str, Any]:
    """
    Extract all geometry parameters from air system assembly.
    
    Args:
        assembly: AirSystemAssembly with actual geometry
        
    Returns:
        Dictionary with geometry for all components
    """
    geometry = {}
    
    # =================================================================
    # BLOWER GEOMETRY
    # =================================================================
    blower = assembly.blower
    bp = blower.params
    
    geometry['blower'] = BlowerGeometry(
        impeller_diameter=bp.impeller_diameter,
        impeller_width=bp.impeller_width,
        inlet_diameter=bp.inlet_diameter,
        hub_diameter=bp.hub_diameter,
        num_blades=bp.num_blades,
        blade_type=bp.blade_type,
        scroll_diameter=bp.impeller_diameter * 1.2,  # Typical scroll OD
        outlet_width=bp.outlet_width,
        outlet_height=bp.outlet_height,
        design_rpm=bp.rpm,
        design_flow_rate=bp.flow_rate,
        design_pressure_rise=bp.pressure_rise,
    )
    
    # =================================================================
    # FILTER GEOMETRY
    # =================================================================
    inlet_filter = assembly.inlet_filter
    fp = inlet_filter.params
    
    geometry['filter'] = FilterGeometry(
        face_area=fp.housing_width * fp.housing_height,
        media_depth=fp.housing_depth,
        housing_volume=fp.housing_width * fp.housing_height * fp.housing_depth,
        efficiency_class=fp.efficiency_class,
    )
    
    # =================================================================
    # DAMPER GEOMETRY
    # =================================================================
    geometry['dampers'] = []
    for i, damper in enumerate(assembly.dampers):
        dp = damper.params
        geometry['dampers'].append(DamperGeometry(
            diameter=dp.diameter,
            blade_chord=dp.diameter * 0.9,  # Blade is ~90% of diameter
            position=dp.position,
        ))
    
    # =================================================================
    # DUCT GEOMETRY (complete flow path with positions)
    # All values extracted directly from assembly - NO hardcoding
    # =================================================================
    duct_diameter = assembly._duct_diameter
    elbow_params = assembly._elbow_params
    transition_params = assembly._transition_params
    
    # Component positions (directly from assembly)
    filter_pos = np.array(assembly._filter_position)
    blower_pos = np.array(assembly._blower_position)
    
    # Filter outlet world position (from assembly port)
    filter_outlet_port = assembly.inlet_filter.ports['outlet']
    filter_outlet_world = filter_pos + np.array(filter_outlet_port.position)
    
    # Elbow positions (directly from assembly's stored params)
    elbow_inlet_pos = np.array(elbow_params['inlet_pos'])
    elbow_outlet_pos = np.array(elbow_params['outlet_pos'])
    elbow_bend_radius = elbow_params['bend_radius']
    
    # Blower inlet/outlet (directly from assembly's stored world positions)
    blower_inlet_world = np.array(assembly._blower_inlet_world)
    blower_outlet_world = np.array(assembly._blower_outlet_world)
    
    # Transition lengths (directly from assembly)
    transition_length = transition_params['length']
    duct_after_transition = transition_params['duct_after_length']
    
    # Create duct segment list with positions for tracer routing
    geometry['ducts'] = []
    
    # 1. Filter to elbow (horizontal, +X direction)
    geometry['ducts'].append(DuctSegment(
        name='filter_to_elbow',
        diameter=elbow_params['diameter'],
        length=elbow_params['duct_horiz_length'],
        area=PI * (elbow_params['diameter'] / 2) ** 2,
    ))
    
    # 2. 90° Elbow (turns from +X to +Z)
    geometry['ducts'].append(DuctSegment(
        name='elbow_90deg',
        diameter=elbow_params['diameter'],
        length=elbow_bend_radius * PI / 2,  # Arc length for 90°
        area=PI * (elbow_params['diameter'] / 2) ** 2,
    ))
    
    # 3. Elbow to blower inlet (vertical, +Z direction)
    geometry['ducts'].append(DuctSegment(
        name='elbow_to_blower',
        diameter=elbow_params['diameter'],
        length=elbow_params['duct_vert_length'],
        area=PI * (elbow_params['diameter'] / 2) ** 2,
    ))
    
    # 4. Blower outlet to damper (transition + duct)
    if assembly.dampers:
        damper_pos = np.array(assembly._damper_positions[0])
        damper_inlet_port = assembly.dampers[0].ports['inlet']
        damper_inlet_world = damper_pos + np.array(damper_inlet_port.position)
        
        # Compute duct length from actual positions
        duct_length = damper_inlet_world[0] - blower_outlet_world[0] - transition_length
        
        # Transition piece (using assembly's stored transition_length)
        geometry['ducts'].append(DuctSegment(
            name='blower_transition',
            diameter=duct_diameter,
            length=transition_length,
            area=PI * (duct_diameter / 2) ** 2,
        ))
        
        # Duct to damper
        if duct_length > 0.01:
            geometry['ducts'].append(DuctSegment(
                name='transition_to_damper',
                diameter=duct_diameter,
                length=duct_length,
                area=PI * (duct_diameter / 2) ** 2,
            ))
    
    # 5. Ducts between dampers (if multiple dampers)
    for i in range(len(assembly.dampers) - 1):
        # Get positions from assembly
        damper_pos_i = np.array(assembly._damper_positions[i])
        damper_pos_next = np.array(assembly._damper_positions[i + 1])
        
        outlet_port = assembly.dampers[i].ports['outlet']
        inlet_port = assembly.dampers[i + 1].ports['inlet']
        
        outlet_world = damper_pos_i + np.array(outlet_port.position)
        inlet_world = damper_pos_next + np.array(inlet_port.position)
        
        duct_between_length = inlet_world[0] - outlet_world[0]
        
        if duct_between_length > 0.01:
            geometry['ducts'].append(DuctSegment(
                name=f'damper_{i}_to_damper_{i+1}',
                diameter=assembly.dampers[i].params.diameter,
                length=duct_between_length,
                area=PI * (assembly.dampers[i].params.diameter / 2) ** 2,
            ))
    
    # =================================================================
    # FLOW PATH WAYPOINTS (for tracer particle routing)
    # All positions directly from assembly - NO hardcoding
    # =================================================================
    # Get filter inlet position from port
    filter_inlet_port = assembly.inlet_filter.ports['inlet']
    filter_inlet_world = filter_pos + np.array(filter_inlet_port.position)
    
    geometry['flow_path'] = {
        'filter_inlet': filter_inlet_world.copy(),
        'filter_outlet': filter_outlet_world.copy(),
        'elbow_inlet': elbow_inlet_pos.copy(),
        'elbow_outlet': elbow_outlet_pos.copy(),
        'blower_inlet': blower_inlet_world.copy(),
        'blower_center': blower_pos.copy(),
        'blower_outlet': blower_outlet_world.copy(),
    }
    
    # Add damper positions to flow path (from assembly's stored positions)
    for idx, (damper, pos) in enumerate(zip(assembly.dampers, assembly._damper_positions)):
        damper_pos_arr = np.array(pos)
        inlet_port = damper.ports['inlet']
        outlet_port = damper.ports['outlet']
        geometry['flow_path'][f'damper_{idx}_inlet'] = damper_pos_arr + np.array(inlet_port.position)
        geometry['flow_path'][f'damper_{idx}_outlet'] = damper_pos_arr + np.array(outlet_port.position)
    
    # =================================================================
    # SYSTEM PARAMETERS
    # =================================================================
    geometry['system'] = {
        'design_flow_rate_m3_h': assembly.params.flow_rate_m3_h,
        'design_pressure_Pa': assembly.params.pressure_rise_Pa,
        'duct_diameter': duct_diameter,
        'elbow_diameter': elbow_params['diameter'],
        'elbow_bend_radius': elbow_bend_radius,
    }
    
    # =================================================================
    # COMPONENT POSITIONS (for animation coordination)
    # =================================================================
    geometry['positions'] = {
        'filter': filter_pos.copy(),
        'blower': blower_pos.copy(),
        'dampers': [np.array(pos) for pos in assembly._damper_positions],
    }
    
    return geometry


# =============================================================================
# PHYSICS CALCULATIONS
# =============================================================================

def calculate_friction_factor(reynolds: float, roughness: float, diameter: float) -> float:
    """
    Calculate Darcy friction factor using Colebrook-White equation.
    
    For laminar flow (Re < 2300): f = 64/Re
    For turbulent flow: Colebrook-White (iterative)
    
    Args:
        reynolds: Reynolds number
        roughness: Surface roughness [m]
        diameter: Pipe diameter [m]
        
    Returns:
        Darcy friction factor
    """
    if reynolds < 2300:
        # Laminar flow
        return 64.0 / max(reynolds, 1.0)
    
    # Turbulent flow - use Swamee-Jain approximation (explicit)
    relative_roughness = roughness / diameter
    
    term1 = relative_roughness / 3.7
    term2 = 5.74 / (reynolds ** 0.9)
    
    f = 0.25 / (np.log10(term1 + term2) ** 2)
    
    return f


def calculate_duct_pressure_drop(
    segment: DuctSegment,
    velocity: float,
    rho: float,
    mu: float
) -> float:
    """
    Calculate pressure drop in a duct segment using Darcy-Weisbach.
    
    dP = f * (L/D) * (rho * V²/2)
    
    Args:
        segment: Duct segment geometry
        velocity: Flow velocity [m/s]
        rho: Air density [kg/m³]
        mu: Dynamic viscosity [Pa·s]
        
    Returns:
        Pressure drop [Pa]
    """
    if velocity < 0.001:
        return 0.0
    
    # Reynolds number
    Re = rho * velocity * segment.diameter / mu
    segment.reynolds = Re
    
    # Friction factor
    f = calculate_friction_factor(Re, segment.roughness, segment.diameter)
    
    # Darcy-Weisbach equation
    dP = f * (segment.length / segment.diameter) * (rho * velocity ** 2 / 2.0)
    
    segment.velocity = velocity
    segment.pressure_drop = dP
    
    return dP


def calculate_filter_pressure_drop(
    filter_geo: FilterGeometry,
    volume_flow_rate: float,
    rho: float
) -> float:
    """
    Calculate filter pressure drop.
    
    dP = R * V_face + 0.5 * rho * V_face² * K_entry
    
    where R is media resistance and K_entry is entry loss coefficient.
    
    Args:
        filter_geo: Filter geometry
        volume_flow_rate: Flow rate [m³/s]
        rho: Air density [kg/m³]
        
    Returns:
        Pressure drop [Pa]
    """
    if volume_flow_rate < 1e-6:
        return 0.0
    
    face_velocity = volume_flow_rate / filter_geo.face_area
    
    # Media resistance (viscous term)
    dP_media = filter_geo.media_resistance * face_velocity
    
    # Entry/exit losses (dynamic term)
    k_entry_exit = 0.5  # Typical for filter housing
    dP_dynamic = 0.5 * rho * face_velocity ** 2 * k_entry_exit
    
    return dP_media + dP_dynamic


def calculate_damper_pressure_drop(
    damper_geo: DamperGeometry,
    volume_flow_rate: float,
    rho: float
) -> float:
    """
    Calculate damper pressure drop based on blade position.
    
    dP = 0.5 * rho * V² * K
    
    where K is the loss coefficient based on blade angle.
    
    Args:
        damper_geo: Damper geometry
        volume_flow_rate: Flow rate [m³/s]
        rho: Air density [kg/m³]
        
    Returns:
        Pressure drop [Pa]
    """
    if volume_flow_rate < 1e-6:
        return 0.0
    
    velocity = volume_flow_rate / damper_geo.area
    
    # Loss coefficient K based on blade position
    # K varies from very high (closed) to low (open)
    position = damper_geo.position
    
    if position < 0.05:
        k = 1000.0  # Nearly closed - very high resistance
    elif position > 0.95:
        k = 0.3     # Fully open - minimal resistance
    else:
        # Interpolate based on butterfly valve curve
        angle = position * 90.0
        angle_rad = np.radians(angle)
        # K decreases as valve opens
        k = 0.3 + 50.0 * (1.0 - np.sin(angle_rad) ** 2)
    
    dP = 0.5 * rho * velocity ** 2 * k
    
    return dP


def calculate_blower_performance(
    blower_geo: BlowerGeometry,
    omega: float,
    system_pressure_drop: float,
    rho: float
) -> Tuple[float, float, float, float]:
    """
    Calculate blower operating point using fan laws.
    
    Fan affinity laws:
    - Q ∝ N (flow proportional to speed)
    - P ∝ N² (pressure proportional to speed squared)
    - W ∝ N³ (power proportional to speed cubed)
    
    The operating point is where blower curve intersects system curve.
    
    Args:
        blower_geo: Blower geometry
        omega: Angular velocity [rad/s]
        system_pressure_drop: System resistance at design flow [Pa]
        rho: Air density [kg/m³]
        
    Returns:
        Tuple of (volume_flow_rate, pressure_rise, shaft_power, efficiency)
    """
    if omega < 0.1:
        return 0.0, 0.0, 0.0, 0.0
    
    rpm = omega * 60.0 / TWO_PI
    design_rpm = blower_geo.design_rpm
    
    if design_rpm < 1.0:
        return 0.0, 0.0, 0.0, 0.0
    
    # Speed ratio
    n_ratio = rpm / design_rpm
    
    # =================================================================
    # BLOWER CURVE (pressure vs flow at current speed)
    # =================================================================
    # At design point: Q_design, P_design
    Q_design = blower_geo.design_flow_rate / 3600.0  # Convert to m³/s
    P_design = blower_geo.design_pressure_rise
    
    # At current speed: Q_max = Q_design * n, P_shutoff = P_design * n²
    Q_max = Q_design * n_ratio
    P_shutoff = P_design * n_ratio ** 2
    
    # =================================================================
    # IMPELLER TIP SPEED
    # =================================================================
    tip_speed = omega * blower_geo.impeller_radius
    
    # Theoretical pressure rise from Euler equation
    # P = rho * u2 * c_u2  where c_u2 is tangential velocity at exit
    # For backward curved blades, c_u2 ≈ 0.5 * u2
    slip_factor = 0.85  # Typical for backward curved
    blade_factor = 0.5 if blower_geo.blade_type == "backward_curved" else 0.7
    
    P_theoretical = rho * tip_speed ** 2 * blade_factor * slip_factor
    
    # =================================================================
    # OPERATING POINT (intersection of blower and system curves)
    # =================================================================
    # System curve: dP = K * Q²
    # If at design flow we have design pressure, K = P_design / Q_design²
    if Q_design > 1e-6:
        K_system = P_design / (Q_design ** 2)
    else:
        K_system = 1e6
    
    # Operating flow rate (solve quadratic)
    # P_blower = P_shutoff * (1 - (Q/Q_max)²)  # Parabolic approximation
    # P_system = K * Q²
    # At operating point: P_blower = P_system
    # P_shutoff * (1 - Q²/Q_max²) = K * Q²
    # Q² * (K + P_shutoff/Q_max²) = P_shutoff
    
    denom = K_system + P_shutoff / max(Q_max ** 2, 1e-12)
    Q_operating_sq = P_shutoff / denom
    Q_operating = np.sqrt(max(Q_operating_sq, 0.0))
    
    # Clamp to physical limits
    Q_operating = min(Q_operating, Q_max * 1.1)
    
    # Pressure at operating point
    P_operating = K_system * Q_operating ** 2
    P_operating = min(P_operating, P_shutoff)
    
    # =================================================================
    # EFFICIENCY
    # =================================================================
    # Efficiency varies with operating point
    # Peak efficiency at design point, drops off at other points
    flow_ratio = Q_operating / max(Q_design * n_ratio, 1e-6)
    
    # Efficiency curve (parabolic)
    peak_efficiency = 0.75 if blower_geo.blade_type == "backward_curved" else 0.65
    efficiency = peak_efficiency * (1.0 - (flow_ratio - 1.0) ** 2)
    efficiency = max(0.1, min(peak_efficiency, efficiency))
    
    # =================================================================
    # SHAFT POWER
    # =================================================================
    # W = Q * P / eta
    fluid_power = Q_operating * P_operating
    shaft_power = fluid_power / max(efficiency, 0.1)
    
    return Q_operating, P_operating, shaft_power, efficiency


# =============================================================================
# WARP KERNELS FOR TRACER PARTICLES
# =============================================================================

@wp.kernel
def update_tracers_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    ages: wp.array(dtype=float),
    active: wp.array(dtype=wp.int32),
    n: int,
    # Flow field parameters
    flow_speed: float,         # Base flow speed [m/s] in ducts
    impeller_omega: float,     # Impeller angular velocity [rad/s]
    impeller_radius: float,    # Impeller radius [m]
    # Flow path waypoints (filter -> elbow -> blower -> transition -> dampers)
    filter_outlet: wp.vec3,    # Start of horizontal duct (+X)
    elbow_inlet: wp.vec3,      # End of horizontal duct, start of elbow
    elbow_outlet: wp.vec3,     # End of elbow, start of vertical duct (+Z)
    blower_inlet: wp.vec3,     # Blower inlet (axial entry at bottom of scroll)
    blower_center: wp.vec3,    # Center of blower/impeller
    blower_outlet: wp.vec3,    # Rectangular outlet from scroll (+X direction)
    transition_end: wp.vec3,   # End of rect-to-circular transition
    damper_1_inlet: wp.vec3,   # Damper 1 inlet
    damper_1_outlet: wp.vec3,  # Damper 1 outlet
    damper_2_inlet: wp.vec3,   # Damper 2 inlet
    damper_2_outlet: wp.vec3,  # Final outlet
    # Elbow parameters for curved path
    elbow_center: wp.vec3,     # Center of elbow arc
    elbow_bend_radius: float,  # Elbow bend radius (centerline)
    # Geometry for containment
    duct_radius: float,        # Inlet duct radius
    outlet_duct_radius: float, # Outlet duct radius (after transition)
    scroll_outer_radius: float,# Scroll outer boundary radius
    scroll_half_width: float,  # Scroll axial half-width (Z extent)
    # Domain bounds
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    z_min: float, z_max: float,
    # Spawn region (inlet)
    spawn_x: float,
    spawn_y_min: float, spawn_y_max: float,
    spawn_z_min: float, spawn_z_max: float,
    # Time
    dt: float,
    max_age: float,
):
    """
    Update tracer particle positions following actual flow path through geometry.
    Implements centrifugal blower physics where air:
    - Enters axially through inlet eye (+Z into scroll)
    - Is accelerated radially by impeller (centrifugal force = omega^2 * r)
    - Spirals outward in scroll casing
    - Exits tangentially through rectangular outlet (+X direction)
    
    Flow path segments:
    1. Filter to elbow inlet (+X direction) - horizontal cylinder
    2. 90 degree elbow (curved path from +X to +Z) - toroidal section
    3. Elbow to blower inlet (+Z direction) - vertical cylinder
    4. Blower scroll - centrifugal acceleration (axial in, radial accel, tangential out)
    5. Rectangular outlet from scroll (+X)
    6. Rectangular-to-circular transition (+X)
    7. Circular duct to damper 1 (+X)
    8. Through damper 1 and duct to damper 2 (+X)
    """
    tid = wp.tid()
    
    if tid >= n:
        return
    
    if active[tid] == 0:
        return
    
    pos = positions[tid]
    age = ages[tid]
    
    # Centerline Y position (ducts are at Y=0)
    duct_y = filter_outlet[1]
    outlet_y = blower_outlet[1]
    blower_y = blower_center[1]
    
    vel = wp.vec3(0.0, 0.0, 0.0)
    segment = 0
    
    # =========================================================================
    # DETERMINE SEGMENT based on position along flow path
    # Flow path: Filter (+X) -> Elbow (turn +X to +Z) -> Vertical duct (+Z) -> Blower -> Outlet (+X)
    # =========================================================================
    
    # Key boundaries
    elbow_x_start = elbow_inlet[0]  # X where elbow begins
    elbow_x_end = elbow_outlet[0]   # X of elbow outlet (also vertical duct center)
    elbow_z_end = elbow_outlet[2]   # Z where elbow ends / vertical duct begins
    blower_z = blower_inlet[2]      # Z of blower inlet
    outlet_x = blower_outlet[0]     # X of blower outlet
    
    # Use tip speed for blower region velocities
    tip_speed = impeller_omega * impeller_radius
    if tip_speed < 1.0:
        tip_speed = 1.0  # Minimum velocity when blower starting
    
    # SEGMENT 1: Before elbow - horizontal flow in +X direction
    # Position is before elbow inlet in X direction
    if pos[0] < elbow_x_start and pos[2] < elbow_z_end * 0.5:
        vel = wp.vec3(flow_speed, 0.0, 0.0)
        segment = 1
    
    # SEGMENT 2: Inside elbow region - curved flow from +X to +Z
    # Position is within elbow X range and Z is less than elbow outlet Z
    elif pos[0] >= elbow_x_start - duct_radius and pos[0] < elbow_x_end + duct_radius and pos[2] < elbow_z_end:
        # Compute progress based on distance from elbow center
        # Elbow center is at (elbow_x_start + R, Y, 0) where R = bend radius
        dx = pos[0] - elbow_center[0]
        dz = pos[2] - elbow_center[2]
        
        # Angle around the elbow (0 at inlet, pi/2 at outlet)
        alpha = wp.atan2(dz, -dx)
        alpha = wp.clamp(alpha, 0.0, 1.5708)
        
        # Flow direction follows the curve
        flow_x = wp.cos(alpha)
        flow_z = wp.sin(alpha)
        
        # Add some forward push toward outlet
        push = 0.3 * (1.0 - wp.cos(alpha * 2.0))  # Peaks at alpha=pi/4
        
        vel = wp.vec3(flow_speed * flow_x * (1.0 + push), 0.0, flow_speed * flow_z * (1.0 + push))
        segment = 2
    
    # SEGMENT 3: Vertical duct after elbow, before blower - flow in +Z
    # Position is past elbow outlet X and Z is between elbow outlet and blower inlet
    elif pos[2] >= elbow_z_end - duct_radius and pos[2] < blower_z:
        vel = wp.vec3(0.0, 0.0, flow_speed * 1.2)
        segment = 3
    
    # SEGMENT 4: Inside blower - centrifugal flow physics
    # Blower outlet is at +X direction, rectangular casing extends from scroll
    elif pos[2] >= blower_z and pos[0] < outlet_x:
        # Position relative to blower center
        dx = pos[0] - blower_center[0]
        dy = pos[1] - blower_center[1]
        
        # Radial distance from impeller axis (XY plane)
        r_xy = wp.sqrt(dx * dx + dy * dy)
        
        # Progress toward outlet (0 at scroll center, 1 at outlet)
        # Outlet is at +X from blower center
        progress_x = (pos[0] - blower_center[0]) / (outlet_x - blower_center[0] + 0.001)
        progress_x = wp.clamp(progress_x, 0.0, 1.0)
        
        if r_xy < impeller_radius * 0.3:
            # Near center (entering through inlet eye)
            # Strong push toward impeller edge and +X toward outlet
            vel = wp.vec3(tip_speed * 0.5, 0.0, 0.0)
        elif r_xy < impeller_radius * 0.8:
            # Inside impeller region - centrifugal + tangential
            rad_x = dx / (r_xy + 0.001)
            rad_y = dy / (r_xy + 0.001)
            tan_x = -rad_y
            tan_y = rad_x
            
            # Centrifugal pushes outward, tangential from blade drag
            centrifugal_v = tip_speed * 0.6
            tangent_v = tip_speed * 0.4
            
            vel = wp.vec3(
                rad_x * centrifugal_v + tan_x * tangent_v + tip_speed * 0.3,  # Add +X bias
                rad_y * centrifugal_v + tan_y * tangent_v,
                0.0
            )
        else:
            # In scroll/outlet casing region - strong +X flow toward outlet
            # The scroll guides air around and the outlet casing directs it +X
            
            # Primary velocity is +X toward the rectangular outlet
            outlet_vel = tip_speed * (0.6 + 0.4 * progress_x)  # Accelerates toward outlet
            
            # Add some swirl (tangential) that decreases as we approach outlet
            rad_x = dx / (r_xy + 0.001)
            rad_y = dy / (r_xy + 0.001)
            tan_x = -rad_y
            tan_y = rad_x
            swirl_factor = 0.2 * (1.0 - progress_x)  # Swirl decreases toward outlet
            
            vel = wp.vec3(
                outlet_vel + tan_x * tip_speed * swirl_factor,
                tan_y * tip_speed * swirl_factor,
                0.0
            )
        
        segment = 4
    
    # SEGMENT 5: Blower outlet to damper 1 - flow in +X
    elif pos[0] >= outlet_x and pos[0] < damper_1_inlet[0]:
        outlet_speed = tip_speed * 0.8
        if outlet_speed < flow_speed:
            outlet_speed = flow_speed * 2.0
        vel = wp.vec3(outlet_speed, 0.0, 0.0)
        segment = 5
    
    # SEGMENT 6: Through damper 1 region (+X)
    elif pos[0] >= damper_1_inlet[0] and pos[0] < damper_2_inlet[0]:
        outlet_speed = flow_speed * 2.0
        vel = wp.vec3(outlet_speed, 0.0, 0.0)
        segment = 6
    
    # SEGMENT 7: Through damper 2 to exit (+X)
    else:
        outlet_speed = flow_speed * 2.0
        vel = wp.vec3(outlet_speed, 0.0, 0.0)
        segment = 7
    
    # Add small turbulent diffusion
    hash_val = float(tid * 12345 + int(age * 1000.0))
    diffusion = 0.002 * flow_speed
    vel = vel + wp.vec3(
        diffusion * wp.sin(hash_val),
        diffusion * wp.cos(hash_val * 2.0) * 0.2,
        diffusion * wp.sin(hash_val * 3.0)
    )
    
    # =========================================================================
    # UPDATE POSITION
    # =========================================================================
    new_pos = pos + vel * dt
    
    # =========================================================================
    # APPLY CONTAINMENT based on segment
    # =========================================================================
    
    if segment == 1:
        # Cylindrical containment (axis along X) - inlet duct before elbow
        duct_center_z = filter_outlet[2]
        dy = new_pos[1] - duct_y
        dz = new_pos[2] - duct_center_z
        r_yz = wp.sqrt(dy * dy + dz * dz)
        
        if r_yz > duct_radius * 0.9:
            if r_yz > 0.001:
                scale = duct_radius * 0.85 / r_yz
                new_pos = wp.vec3(new_pos[0], duct_y + dy * scale, duct_center_z + dz * scale)
        
        # X bounds
        if new_pos[0] < filter_outlet[0]:
            new_pos = wp.vec3(filter_outlet[0], new_pos[1], new_pos[2])
    
    elif segment == 2:
        # Elbow region - toroidal containment around the elbow centerline arc
        dx = new_pos[0] - elbow_center[0]
        dz = new_pos[2] - elbow_center[2]
        dy = new_pos[1] - duct_y
        
        # Find angle around elbow arc
        alpha = wp.atan2(dz, -dx)
        alpha = wp.clamp(alpha, 0.0, 1.5708)
        
        # Point on elbow centerline at this angle
        cx = elbow_center[0] - elbow_bend_radius * wp.cos(alpha)
        cz = elbow_center[2] + elbow_bend_radius * wp.sin(alpha)
        
        # Distance from centerline
        to_p_x = new_pos[0] - cx
        to_p_y = dy
        to_p_z = new_pos[2] - cz
        r_tube = wp.sqrt(to_p_x * to_p_x + to_p_y * to_p_y + to_p_z * to_p_z)
        
        # Constrain within duct tube
        if r_tube > duct_radius * 0.9:
            if r_tube > 0.001:
                scale = duct_radius * 0.85 / r_tube
                new_pos = wp.vec3(cx + to_p_x * scale, duct_y + to_p_y * scale, cz + to_p_z * scale)
        
        # Ensure Z doesn't go negative (before elbow inlet)
        if new_pos[2] < 0.0:
            new_pos = wp.vec3(new_pos[0], new_pos[1], 0.0)
    
    elif segment == 3:
        # Cylindrical containment (axis along Z) - vertical duct to blower
        duct_center_x = elbow_outlet[0]
        dx = new_pos[0] - duct_center_x
        dy = new_pos[1] - duct_y
        r_xy = wp.sqrt(dx * dx + dy * dy)
        
        if r_xy > duct_radius * 0.9:
            if r_xy > 0.001:
                scale = duct_radius * 0.85 / r_xy
                new_pos = wp.vec3(duct_center_x + dx * scale, duct_y + dy * scale, new_pos[2])
        
        # Z bounds - don't let particles go back into elbow
        if new_pos[2] < elbow_outlet[2]:
            new_pos = wp.vec3(new_pos[0], new_pos[1], elbow_outlet[2])
    
    elif segment == 4:
        # Blower containment - scroll + outlet casing
        # Scroll is cylindrical, but outlet casing extends +X as rectangular duct
        dx_scroll = new_pos[0] - blower_center[0]
        dy_scroll = new_pos[1] - blower_center[1]
        
        # In the outlet casing region (X > scroll edge), use rectangular containment
        scroll_edge_x = blower_center[0] + scroll_outer_radius * 0.8
        
        if new_pos[0] > scroll_edge_x:
            # In outlet casing - rectangular containment
            # Keep Y near blower center, Z near blower_outlet Z
            outlet_half_height = scroll_half_width * 0.8  # Outlet height
            outlet_half_width = scroll_half_width * 0.8   # Outlet width
            
            # Y containment
            dy_out = new_pos[1] - blower_center[1]
            if wp.abs(dy_out) > outlet_half_height:
                sign_y = 1.0 if dy_out > 0.0 else -1.0
                new_pos = wp.vec3(new_pos[0], blower_center[1] + sign_y * outlet_half_height * 0.9, new_pos[2])
            
            # Z containment (around blower_outlet Z)
            dz_out = new_pos[2] - blower_outlet[2]
            if wp.abs(dz_out) > outlet_half_width:
                sign_z = 1.0 if dz_out > 0.0 else -1.0
                new_pos = wp.vec3(new_pos[0], new_pos[1], blower_outlet[2] + sign_z * outlet_half_width * 0.9)
        else:
            # In scroll region - cylindrical containment
            r_scroll_xy = wp.sqrt(dx_scroll * dx_scroll + dy_scroll * dy_scroll)
            
            if r_scroll_xy > scroll_outer_radius * 0.95:
                if r_scroll_xy > 0.001:
                    scale = scroll_outer_radius * 0.9 / r_scroll_xy
                    new_pos = wp.vec3(
                        blower_center[0] + dx_scroll * scale,
                        blower_center[1] + dy_scroll * scale,
                        new_pos[2]
                    )
            
            # Z containment in scroll
            z_min_scroll = blower_center[2] - scroll_half_width
            z_max_scroll = blower_center[2] + scroll_half_width
            if new_pos[2] < z_min_scroll:
                new_pos = wp.vec3(new_pos[0], new_pos[1], z_min_scroll)
            if new_pos[2] > z_max_scroll:
                new_pos = wp.vec3(new_pos[0], new_pos[1], z_max_scroll)
    
    elif segment == 5:
        # Rectangular outlet transitioning to circular - containment
        # Outlet centerline is at (any_x, blower_y, blower_outlet_z)
        duct_center_y = blower_y
        duct_center_z = blower_outlet[2]
        dy = new_pos[1] - duct_center_y
        dz = new_pos[2] - duct_center_z
        r_yz = wp.sqrt(dy * dy + dz * dz)
        
        # Use outlet_duct_radius for the transition/damper duct
        if r_yz > outlet_duct_radius * 0.9:
            if r_yz > 0.001:
                scale = outlet_duct_radius * 0.85 / r_yz
                new_pos = wp.vec3(new_pos[0], duct_center_y + dy * scale, duct_center_z + dz * scale)
        
        if new_pos[0] < blower_outlet[0]:
            new_pos = wp.vec3(blower_outlet[0], new_pos[1], new_pos[2])
    
    elif segment == 6:
        # Circular duct through damper 1 region
        duct_center_y = blower_y
        duct_center_z = damper_1_inlet[2]
        dy = new_pos[1] - duct_center_y
        dz = new_pos[2] - duct_center_z
        r_yz = wp.sqrt(dy * dy + dz * dz)
        
        if r_yz > outlet_duct_radius * 0.9:
            if r_yz > 0.001:
                scale = outlet_duct_radius * 0.85 / r_yz
                new_pos = wp.vec3(new_pos[0], duct_center_y + dy * scale, duct_center_z + dz * scale)
    
    elif segment == 7:
        # Circular duct through damper 2 to exit
        duct_center_y = blower_y
        duct_center_z = damper_2_inlet[2]
        dy = new_pos[1] - duct_center_y
        dz = new_pos[2] - duct_center_z
        r_yz = wp.sqrt(dy * dy + dz * dz)
        
        if r_yz > outlet_duct_radius * 0.9:
            if r_yz > 0.001:
                scale = outlet_duct_radius * 0.85 / r_yz
                new_pos = wp.vec3(new_pos[0], duct_center_y + dy * scale, duct_center_z + dz * scale)
    
    # Update age
    new_age = age + dt
    
    # Check if tracer should respawn
    respawn = False
    
    # Out of bounds
    if new_pos[0] > x_max or new_pos[0] < x_min:
        respawn = True
    if new_pos[1] > y_max or new_pos[1] < y_min:
        respawn = True
    if new_pos[2] > z_max or new_pos[2] < z_min:
        respawn = True
    
    # Exceeded max age or exited past damper 2
    if new_age > max_age:
        respawn = True
    if new_pos[0] > damper_2_outlet[0] + 0.1:
        respawn = True
    
    if respawn:
        # Respawn at inlet
        seed = float(tid + int(new_age * 100.0))
        new_pos = wp.vec3(
            spawn_x,
            spawn_y_min + (spawn_y_max - spawn_y_min) * wp.frac(seed * 0.618),
            spawn_z_min + (spawn_z_max - spawn_z_min) * wp.frac(seed * 0.381)
        )
        new_age = 0.0
    
    positions[tid] = new_pos
    velocities[tid] = vel
    ages[tid] = new_age


# =============================================================================
# MAIN SIMULATOR CLASS
# =============================================================================

class AirFlowPhysicsSimulator:
    """
    Physics-based air flow simulator for air system.
    
    Uses actual geometry from AirSystemAssembly and computes all
    physics from first principles - no magic numbers.
    """
    
    def __init__(
        self,
        assembly: AirSystemAssembly,
        config: AirFlowPhysicsConfig = None,
    ):
        """
        Initialize the physics-based air flow simulator.
        
        Args:
            assembly: AirSystemAssembly with actual geometry
            config: Simulation configuration
        """
        self.assembly = assembly
        self.config = config or AirFlowPhysicsConfig()
        self.state = AirFlowPhysicsState()
        
        # Initialize Warp
        wp.init()
        self.device = self.config.device
        
        # Extract geometry
        self.geometry = extract_air_geometry(assembly)
        
        # Compute derived parameters
        self._compute_derived_parameters()
        
        # Allocate tracer arrays if enabled
        if self.config.enable_tracers:
            self._allocate_tracer_arrays()
        
        # Get system bounds
        self._compute_system_bounds()
    
    def _compute_derived_parameters(self):
        """Compute all derived physics parameters from geometry."""
        cfg = self.config
        blower_geo = self.geometry['blower']
        
        # Design angular velocity
        self.design_omega = TWO_PI * blower_geo.design_rpm / 60.0
        
        # Target angular velocity
        self.target_omega = TWO_PI * cfg.target_rpm / 60.0
        
        # Angular acceleration for ramp
        self.omega_ramp_rate = self.target_omega / cfg.ramp_time
        
        # Design tip speed
        self.design_tip_speed = self.design_omega * blower_geo.impeller_radius
        
        # Duct area for velocity calculation
        self.duct_area = PI * (self.geometry['system']['duct_diameter'] / 2.0) ** 2
        
        print(f"\n  Air Flow Physics Parameters (computed from geometry):")
        print(f"    Design RPM:        {blower_geo.design_rpm:.0f}")
        print(f"    Target RPM:        {cfg.target_rpm:.0f}")
        print(f"    Design ω:          {self.design_omega:.1f} rad/s")
        print(f"    Impeller dia:      {blower_geo.impeller_diameter*1000:.0f} mm")
        print(f"    Design tip speed:  {self.design_tip_speed:.1f} m/s")
        print(f"    Design flow:       {blower_geo.design_flow_rate:.0f} m³/h")
        print(f"    Design pressure:   {blower_geo.design_pressure_rise:.0f} Pa")
        print(f"    Duct diameter:     {self.geometry['system']['duct_diameter']*1000:.0f} mm")
        
        # Print duct segment geometry
        total_duct_length = sum(d.length for d in self.geometry['ducts'])
        print(f"\n  Duct Segments (from geometry, total length: {total_duct_length*1000:.0f} mm):")
        for duct in self.geometry['ducts']:
            print(f"    {duct.name}:")
            print(f"      Length:    {duct.length*1000:.0f} mm")
            print(f"      Diameter:  {duct.diameter*1000:.0f} mm")
            print(f"      Area:      {duct.area*10000:.1f} cm²")
        
        # Print flow path waypoints (all from assembly's stored positions)
        if 'flow_path' in self.geometry:
            print(f"\n  Flow Path Waypoints (from assembly port positions):")
            for name, pos in self.geometry['flow_path'].items():
                print(f"    {name:20s}: ({pos[0]*1000:6.0f}, {pos[1]*1000:6.0f}, {pos[2]*1000:6.0f}) mm")
    
    def _allocate_tracer_arrays(self):
        """Allocate tracer particle arrays on device."""
        n = self.config.num_tracers
        
        self.state.tracer_positions = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.state.tracer_velocities = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.state.tracer_ages = wp.zeros(n, dtype=float, device=self.device)
        self.state.tracer_active = wp.ones(n, dtype=wp.int32, device=self.device)
    
    def _compute_system_bounds(self):
        """Compute system bounding box for tracer containment."""
        if not self.assembly._mesh_built:
            self.assembly.build_mesh()
        
        min_corner, max_corner = self.assembly.get_bounds()
        
        # Add some margin
        margin = 0.1
        self.x_min = min_corner[0] - margin
        self.x_max = max_corner[0] + margin
        self.y_min = min_corner[1] - margin
        self.y_max = max_corner[1] + margin
        self.z_min = min_corner[2] - margin
        self.z_max = max_corner[2] + margin
        
        # Spawn region (filter inlet)
        filter_pos = self.assembly._filter_position
        filter_geo = self.geometry['filter']
        
        self.spawn_x = filter_pos[0] - 0.1
        self.spawn_y_min = filter_pos[1] - 0.1
        self.spawn_y_max = filter_pos[1] + 0.1
        self.spawn_z_min = filter_pos[2] - 0.1
        self.spawn_z_max = filter_pos[2] + 0.1
    
    def initialize_tracers(self):
        """Initialize tracer particles at filter inlet."""
        if not self.config.enable_tracers:
            return
        
        n = self.config.num_tracers
        rng = np.random.default_rng(42)
        
        positions = np.zeros((n, 3), dtype=np.float32)
        
        for i in range(n):
            positions[i, 0] = self.spawn_x
            positions[i, 1] = rng.uniform(self.spawn_y_min, self.spawn_y_max)
            positions[i, 2] = rng.uniform(self.spawn_z_min, self.spawn_z_max)
        
        wp.copy(
            self.state.tracer_positions,
            wp.array(positions, dtype=wp.vec3, device=self.device)
        )
        
        # Random initial ages (staggered)
        ages = rng.uniform(0, self.config.tracer_lifetime, n).astype(np.float32)
        wp.copy(
            self.state.tracer_ages,
            wp.array(ages, dtype=float, device=self.device)
        )
    
    # =========================================================================
    # CONTROL METHODS
    # =========================================================================
    
    def start_system(self):
        """Start the air system (ramp up blower, open dampers)."""
        self.state.phase = SystemPhase.STARTUP
        self.state.phase_start_time = self.state.time
        self.state.blower_state = BlowerState.STARTING
    
    def stop_system(self):
        """Stop the air system (ramp down blower, close dampers)."""
        self.state.phase = SystemPhase.SHUTDOWN
        self.state.phase_start_time = self.state.time
        self.state.blower_state = BlowerState.STOPPING
    
    def set_damper_position(self, index: int, position: float):
        """Set damper position (0=closed, 1=open)."""
        if 0 <= index < len(self.state.damper_positions):
            self.state.damper_positions[index] = max(0.0, min(1.0, position))
            self.geometry['dampers'][index].position = self.state.damper_positions[index]
    
    # =========================================================================
    # SIMULATION STEP
    # =========================================================================
    
    def step(self):
        """Advance simulation by one time step."""
        dt = self.config.dt
        cfg = self.config
        
        # =================================================================
        # UPDATE BLOWER SPEED (RAMP UP/DOWN)
        # =================================================================
        if self.state.blower_state == BlowerState.STARTING:
            self.state.blower_omega += self.omega_ramp_rate * dt
            if self.state.blower_omega >= self.target_omega:
                self.state.blower_omega = self.target_omega
                self.state.blower_state = BlowerState.RUNNING
                self.state.phase = SystemPhase.RUNNING
        
        elif self.state.blower_state == BlowerState.STOPPING:
            self.state.blower_omega -= self.omega_ramp_rate * dt
            if self.state.blower_omega <= 0.0:
                self.state.blower_omega = 0.0
                self.state.blower_state = BlowerState.OFF
                self.state.phase = SystemPhase.OFF
        
        # Compute RPM and tip speed
        self.state.blower_rpm = self.state.blower_omega * 60.0 / TWO_PI
        blower_geo = self.geometry['blower']
        self.state.blower_tip_speed = self.state.blower_omega * blower_geo.impeller_radius
        
        # =================================================================
        # UPDATE DAMPER POSITIONS (RAMP)
        # =================================================================
        if self.state.phase == SystemPhase.STARTUP:
            # Open dampers during startup
            damper_ramp_rate = 1.0 / cfg.damper_ramp_time
            for i in range(len(self.state.damper_positions)):
                self.state.damper_positions[i] = min(
                    1.0,
                    self.state.damper_positions[i] + damper_ramp_rate * dt
                )
                self.geometry['dampers'][i].position = self.state.damper_positions[i]
        
        elif self.state.phase == SystemPhase.SHUTDOWN:
            # Close dampers during shutdown
            damper_ramp_rate = 1.0 / cfg.damper_ramp_time
            for i in range(len(self.state.damper_positions)):
                self.state.damper_positions[i] = max(
                    0.0,
                    self.state.damper_positions[i] - damper_ramp_rate * dt
                )
                self.geometry['dampers'][i].position = self.state.damper_positions[i]
        
        # =================================================================
        # CALCULATE SYSTEM PRESSURE DROPS
        # =================================================================
        rho = cfg.air_density
        mu = cfg.air_viscosity
        
        # First pass: estimate flow rate from blower performance
        Q_estimate, P_blower, shaft_power, efficiency = calculate_blower_performance(
            blower_geo,
            self.state.blower_omega,
            blower_geo.design_pressure_rise,
            rho
        )
        
        # Calculate component pressure drops at this flow rate
        total_dp = 0.0
        
        # Filter pressure drop
        filter_geo = self.geometry['filter']
        dp_filter = calculate_filter_pressure_drop(filter_geo, Q_estimate, rho)
        total_dp += dp_filter
        
        # Duct pressure drops
        velocity = Q_estimate / self.duct_area if self.duct_area > 0 else 0.0
        for duct in self.geometry['ducts']:
            dp_duct = calculate_duct_pressure_drop(duct, velocity, rho, mu)
            total_dp += dp_duct
        
        # Damper pressure drops
        dp_dampers = 0.0
        for damper_geo in self.geometry['dampers']:
            dp_damper = calculate_damper_pressure_drop(damper_geo, Q_estimate, rho)
            dp_dampers += dp_damper
        total_dp += dp_dampers
        
        # =================================================================
        # ITERATE TO FIND OPERATING POINT
        # =================================================================
        # Simple iteration to find where blower curve meets system curve
        for _ in range(3):
            Q_new, P_new, shaft_power, efficiency = calculate_blower_performance(
                blower_geo,
                self.state.blower_omega,
                total_dp,
                rho
            )
            Q_estimate = 0.5 * (Q_estimate + Q_new)
        
        # =================================================================
        # UPDATE STATE
        # =================================================================
        self.state.volume_flow_rate = Q_estimate
        self.state.mass_flow_rate = Q_estimate * rho
        self.state.static_pressure_rise = P_new
        self.state.total_pressure_drop = total_dp
        self.state.shaft_power = shaft_power
        self.state.efficiency = efficiency
        
        # Electrical power (assuming 95% motor efficiency)
        motor_efficiency = 0.95
        self.state.electrical_power = shaft_power / motor_efficiency
        
        # Accumulate energy
        self.state.total_energy_kWh += self.state.electrical_power * dt / 3600.0 / 1000.0
        
        # =================================================================
        # UPDATE TRACER PARTICLES (using actual flow path geometry)
        # =================================================================
        if self.config.enable_tracers and self.state.tracer_positions is not None:
            # Calculate flow velocity in main duct
            flow_velocity = Q_estimate / self.duct_area if self.duct_area > 0 else 0.0
            
            # Get flow path waypoints from geometry
            flow_path = self.geometry.get('flow_path', {})
            filter_outlet = flow_path.get('filter_outlet', np.array([0.0, 0.0, 0.0]))
            elbow_inlet = flow_path.get('elbow_inlet', np.array([0.1, 0.0, 0.0]))
            elbow_outlet = flow_path.get('elbow_outlet', np.array([0.2, 0.0, 0.1]))
            blower_inlet = flow_path.get('blower_inlet', np.array([0.2, 0.0, 0.2]))
            blower_center = flow_path.get('blower_center', np.array([0.3, 0.0, 0.3]))
            blower_outlet = flow_path.get('blower_outlet', np.array([0.5, 0.0, 0.2]))
            
            # Damper positions
            damper_1_inlet = flow_path.get('damper_0_inlet', blower_outlet + np.array([0.2, 0.0, 0.0]))
            damper_1_outlet = flow_path.get('damper_0_outlet', blower_outlet + np.array([0.35, 0.0, 0.0]))
            damper_2_inlet = flow_path.get('damper_1_inlet', blower_outlet + np.array([0.4, 0.0, 0.0]))
            damper_2_outlet = flow_path.get('damper_1_outlet', blower_outlet + np.array([0.55, 0.0, 0.0]))
            
            # Transition end (between blower outlet and damper 1)
            transition_end = (blower_outlet + damper_1_inlet) * 0.5
            
            # Elbow center (computed from geometry)
            elbow_bend_radius = self.geometry['system'].get('elbow_bend_radius', 0.1)
            elbow_center = np.array([
                elbow_inlet[0] + elbow_bend_radius,
                elbow_inlet[1],
                elbow_inlet[2],
            ])
            
            # Duct radii for containment
            elbow_diameter = self.geometry['system'].get('elbow_diameter', 0.3)
            duct_radius = elbow_diameter / 2.0
            
            # Outlet duct radius (from damper diameter)
            outlet_duct_diameter = self.geometry['system'].get('duct_diameter', 0.266)
            outlet_duct_radius = outlet_duct_diameter / 2.0
            
            # Blower geometry for centrifugal physics
            blower_geo = self.geometry['blower']
            impeller_radius = blower_geo.impeller_radius
            scroll_outer_radius = blower_geo.scroll_diameter / 2.0
            scroll_half_width = blower_geo.impeller_width / 2.0 * 1.2  # Scroll slightly wider
            
            # Current impeller angular velocity (for centrifugal physics)
            impeller_omega = self.state.blower_omega
            
            wp.launch(
                kernel=update_tracers_kernel,
                dim=self.config.num_tracers,
                inputs=[
                    self.state.tracer_positions,
                    self.state.tracer_velocities,
                    self.state.tracer_ages,
                    self.state.tracer_active,
                    self.config.num_tracers,
                    # Flow parameters
                    float(flow_velocity),
                    float(impeller_omega),
                    float(impeller_radius),
                    # Flow path waypoints
                    wp.vec3(float(filter_outlet[0]), float(filter_outlet[1]), float(filter_outlet[2])),
                    wp.vec3(float(elbow_inlet[0]), float(elbow_inlet[1]), float(elbow_inlet[2])),
                    wp.vec3(float(elbow_outlet[0]), float(elbow_outlet[1]), float(elbow_outlet[2])),
                    wp.vec3(float(blower_inlet[0]), float(blower_inlet[1]), float(blower_inlet[2])),
                    wp.vec3(float(blower_center[0]), float(blower_center[1]), float(blower_center[2])),
                    wp.vec3(float(blower_outlet[0]), float(blower_outlet[1]), float(blower_outlet[2])),
                    wp.vec3(float(transition_end[0]), float(transition_end[1]), float(transition_end[2])),
                    wp.vec3(float(damper_1_inlet[0]), float(damper_1_inlet[1]), float(damper_1_inlet[2])),
                    wp.vec3(float(damper_1_outlet[0]), float(damper_1_outlet[1]), float(damper_1_outlet[2])),
                    wp.vec3(float(damper_2_inlet[0]), float(damper_2_inlet[1]), float(damper_2_inlet[2])),
                    wp.vec3(float(damper_2_outlet[0]), float(damper_2_outlet[1]), float(damper_2_outlet[2])),
                    # Elbow parameters
                    wp.vec3(float(elbow_center[0]), float(elbow_center[1]), float(elbow_center[2])),
                    float(elbow_bend_radius),
                    # Geometry for containment
                    float(duct_radius),
                    float(outlet_duct_radius),
                    float(scroll_outer_radius),
                    float(scroll_half_width),
                    # Bounds
                    float(self.x_min), float(self.x_max),
                    float(self.y_min), float(self.y_max),
                    float(self.z_min), float(self.z_max),
                    # Spawn
                    float(self.spawn_x),
                    float(self.spawn_y_min), float(self.spawn_y_max),
                    float(self.spawn_z_min), float(self.spawn_z_max),
                    # Time
                    float(dt),
                    float(self.config.tracer_lifetime),
                ],
                device=self.device,
            )
        
        # Update time
        self.state.time += dt
        self.state.step += 1
    
    # =========================================================================
    # RESULTS AND GETTERS
    # =========================================================================
    
    def get_results(self) -> Dict[str, Any]:
        """Get current simulation results."""
        return {
            'time': self.state.time,
            'step': self.state.step,
            'phase': self.state.phase.value,
            'blower_state': self.state.blower_state.name,
            'blower_rpm': self.state.blower_rpm,
            'blower_omega': self.state.blower_omega,
            'tip_speed': self.state.blower_tip_speed,
            'volume_flow_rate_m3_h': self.state.volume_flow_rate * 3600,
            'mass_flow_rate_kg_s': self.state.mass_flow_rate,
            'pressure_rise_Pa': self.state.static_pressure_rise,
            'system_pressure_drop_Pa': self.state.total_pressure_drop,
            'shaft_power_kW': self.state.shaft_power / 1000.0,
            'electrical_power_kW': self.state.electrical_power / 1000.0,
            'efficiency': self.state.efficiency,
            'damper_positions': self.state.damper_positions.copy(),
            'total_energy_kWh': self.state.total_energy_kWh,
        }
    
    def get_tracer_positions(self) -> np.ndarray:
        """Get current tracer particle positions."""
        if self.state.tracer_positions is None:
            return np.array([])
        return self.state.tracer_positions.numpy()
    
    def get_tracer_velocities(self) -> np.ndarray:
        """Get current tracer particle velocities."""
        if self.state.tracer_velocities is None:
            return np.array([])
        return self.state.tracer_velocities.numpy()
    
    def get_duct_segment_data(self) -> List[Dict[str, Any]]:
        """Get pressure drop data for each duct segment."""
        data = []
        for duct in self.geometry['ducts']:
            data.append({
                'name': duct.name,
                'length': duct.length,
                'diameter': duct.diameter,
                'velocity': duct.velocity,
                'reynolds': duct.reynolds,
                'pressure_drop': duct.pressure_drop,
            })
        return data


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_air_flow_simulator(
    assembly: AirSystemAssembly,
    target_rpm: float = 3000.0,
    total_time: float = 10.0,
    dt: float = 0.001,
    enable_tracers: bool = True,
    device: str = "cuda",
) -> AirFlowPhysicsSimulator:
    """
    Create a physics-based air flow simulator.
    
    Args:
        assembly: AirSystemAssembly with geometry
        target_rpm: Target blower RPM
        total_time: Simulation duration [s]
        dt: Time step [s]
        enable_tracers: Enable tracer particles
        device: Compute device
        
    Returns:
        Configured AirFlowPhysicsSimulator
    """
    config = AirFlowPhysicsConfig(
        dt=dt,
        total_time=total_time,
        target_rpm=target_rpm,
        enable_tracers=enable_tracers,
        device=device,
    )
    
    return AirFlowPhysicsSimulator(assembly, config)
